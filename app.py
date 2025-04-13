import streamlit as st
import PyPDF2
import io
from ingestion import process_pdf
from ingestion import clear_database
from query import query_collection, query_llm_with_context

def main():
    st.title("PDF Uploader")
    st.write("Upload a PDF file to view its content")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Display file details
        file_details = {"Filename": uploaded_file.name, "File size": uploaded_file.size}
        st.write(file_details)
        
        # Read PDF content
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            num_pages = len(pdf_reader.pages)
            st.write(f"Number of pages: {num_pages}")
            
            # Display text from the first few pages
            text = ""
            for page_num in range(min(5, num_pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            st.subheader("PDF Content (first few pages)")
            st.text_area("Text", text, height=300)
            
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process PDF"):
                process_pdf(pdf_reader, collection_name="documents", chunk_size=1000)
                # Display success message
                st.success("PDF processed and stored in Qdrant database.")
                # Set a session state flag to indicate processing is done
                st.session_state.pdf_processed = True
        
        with col2:
            if st.button("Clear Database"):
                clear_database()
                st.success("Database cleared successfully.")
                # Reset the session state
                if 'pdf_processed' in st.session_state:
                    st.session_state.pdf_processed = False
    
    st.divider()
    st.subheader("Query Your Documents")
    
    # Text input for user query
    query = st.text_input("Enter your question about the document:")
    
    if st.button("Search") and query:
        with st.spinner("Searching for relevant information..."):
            # Retrieve context from the database
            search_results = query_collection(collection_name="documents", query_text=query, limit=3)
            
            if search_results:
                # Extract text from the results to use as context
                context = "\n\n".join([result.get("text", "") for result in search_results])
                
                with st.spinner("Generating response based on the document..."):
                    # Get LLM response
                    response = query_llm_with_context(context, query)
                    
                    # Display the response
                    st.subheader("Response")
                    # Reasoning is denoted within <think> and </think> tags
                    reasoning = response.split("<think>")[1].split("</think>")[0] if "<think>" in response else ""
                    final_answer = response.replace(f"<think>{reasoning}</think>", "").strip()
                    st.write(final_answer)
                    
                    # Optionally show the supporting evidence
                    with st.expander("View supporting context"):
                        st.write(context)
            else:
                st.warning("No relevant information found in the document. Try a different question or process more documents.")

if __name__ == "__main__":
    main()