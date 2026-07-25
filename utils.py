import fitz

def read_pdf(pdf):

    text = ""

    document = fitz.open(stream=pdf.read(), filetype="pdf")

    for page in document:

        text += page.get_text()

    return text
