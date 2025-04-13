import streamlit as st
import PyPDF2
import io
from ingestion import process_pdf
from ingestion import clear_database
from query import query_collection, query_llm_with_context

def main():
    st.title("PDF Uploader")
    st.write("Upload a PDF file to view its content")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    
    if uploaded_file:
        st.info(f"📄 {len(uploaded_file)} file(s) uploaded")
        for file in uploaded_file:
            # Display file details
            st.subheader(f"File: {file.name}")
            file_details = {"Filename": file.name, "File size": file.size}
            st.write(file_details)
            
            # Read PDF content
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                num_pages = len(pdf_reader.pages)
                st.write(f"Number of pages: {num_pages}")
                
                # Display text from the first few pages
                text = ""
                for page_num in range(min(5, num_pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                st.subheader("PDF Content (first few pages)")
                st.text_area(f"Text from {file.name}", text, height=300)
                
                # Reset file pointer for processing
                file.seek(0)
                
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        
        # Process and clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process PDFs"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Callback function to update progress
                def update_progress(progress_value, status):
                    progress_bar.progress(progress_value)
                    status_text.text(f"{status}: {int(progress_value * 100)}% complete")
                
                try:
                    total_chunks = 0
                    # Process each PDF
                    for file in uploaded_file:
                        file.seek(0)  # Reset file pointer
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                        num_chunks = process_pdf(pdf_reader, collection_name="documents", progress_callback=update_progress)
                        total_chunks += num_chunks
                    
                    # Complete the progress
                    progress_bar.progress(1.0)
                    status_text.text(f"Processing complete! Processed {total_chunks} text chunks from {len(uploaded_file)} files.")
                    
                    # Display success message
                    st.success("All PDFs processed and stored in Qdrant database.")
                    # Set a session state flag to indicate processing is done
                    st.session_state.pdf_processed = True
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")
                    progress_bar.empty()
                    status_text.empty()
        
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
    query = st.text_area("Enter your question about the document:")
    
    if st.button("Search") and query:
        with st.spinner("Searching for relevant information..."):
            # Retrieve context from the database
            search_results = query_collection(query_text=query)
            
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