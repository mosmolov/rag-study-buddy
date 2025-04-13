import streamlit as st
import PyPDF2
import io
from ingestion import process_pdf
from ingestion import clear_database
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
        if st.button("Process PDF"):
            process_pdf(pdf_reader, collection_name="documents", chunk_size=1000)
            # Display success message
            st.success("PDF processed and stored in Qdrant database.")
            # Add a button to clear the database
            if st.button("Clear Database"):
                clear_database()
                st.success("Database cleared successfully.")
if __name__ == "__main__":
    main()