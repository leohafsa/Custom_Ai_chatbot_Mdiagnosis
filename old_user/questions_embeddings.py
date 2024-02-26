from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# Path to your text file
q_document_path = 'C:\\Users\\EZShifa\\Desktop\\chatbot_api\\old_user\\question.txt'

sources = []
with open(q_document_path, 'r') as file:
    for line in file:
        content= line.strip().split(',')  # Assuming content and diagnosis are separated by a comma
        print("content :",content)
#         doc = Document(
#             page_content=content,
#             metadata={"source":},
#         )
#         sources.append(doc)

# chunks = []
# splitter = RecursiveCharacterTextSplitter(
#     separators=["\n", ".", "!", "?", ",", " ", "<br>"],
#     chunk_size=200,
#     chunk_overlap=0
# )

# for source in sources:
#     for chunk in splitter.split_text(source.page_content):
#         chunks.append(Document(page_content=chunk, metadata=source.metadata))

# openai_api_key = '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'
# index_object = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))
