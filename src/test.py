from docx import Document

def read_docx(path):
    doc = Document(path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Test
text = read_docx("docs/RAG.docx")
print(text[:500])  # affiche les 500 premiers caractères
