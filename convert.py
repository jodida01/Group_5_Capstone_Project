import win32com.client as win32
import os
import glob

def convert_doc_to_docx(doc_path, docx_path=None):
    word = win32.Dispatch("Word.Application")
    word.Visible = False  # Run in background

    doc = word.Documents.Open(os.path.abspath(doc_path))

    if docx_path is None:
        docx_path = os.path.splitext(doc_path)[0] + ".docx"

    doc.SaveAs(os.path.abspath(docx_path), FileFormat=16)

    doc.Close()
    word.Quit()

    print(f"✅ Converted: {doc_path} → {docx_path}")

# 🔄 Convert all .doc files in current folder
for file in glob.glob("*.doc"):cl