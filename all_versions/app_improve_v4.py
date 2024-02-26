import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.vectorstores.faiss import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request,jsonify
import openai
import time
import os
api_key = "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"
model_name = "gpt-3.5-turbo"
medicine_list = [
    {"symptoms": "Headache", "prescribed_medicine": "Acetaminophen", "dosage": "500mg-1000mg every 4-6 hours"},
    {"symptoms": "Fever", "prescribed_medicine": "Ibuprofen", "dosage": "200mg-400mg every 4-6 hours with food"},
    {"symptoms": "Allergies", "prescribed_medicine": "Loratadine", "dosage": "10mg once daily"},
    {"symptoms": "Cough", "prescribed_medicine": "Dextromethorphan", "dosage": "10-20mg every 4 hours"},
    {"symptoms": "Common Cold", "prescribed_medicine": "Pseudoephedrine", "dosage": "60mg every 4-6 hours"},
    {"symptoms": "Acid Reflux", "prescribed_medicine": "Omeprazole", "dosage": "20mg once daily before a meal"},
    {"symptoms": "Insomnia", "prescribed_medicine": "Zolpidem", "dosage": "5mg-10mg before bedtime"},
    {"symptoms": "High Blood Pressure", "prescribed_medicine": "Lisinopril", "dosage": "10mg once daily"},
    {"symptoms": "Nausea", "prescribed_medicine": "Ondansetron", "dosage": "4mg-8mg every 8 hours as needed"},
    {"symptoms": "Diarrhea", "prescribed_medicine": "Loperamide", "dosage": "4mg initially, then 2mg after each loose stool"},
    {"symptoms": "Migraine", "prescribed_medicine": "Sumatriptan", "dosage": "25mg-100mg at the onset of migraine symptoms"},
    {"symptoms": "Anxiety", "prescribed_medicine": "Alprazolam", "dosage": "0.25mg-0.5mg three times a day as needed"},
    {"symptoms": "Depression", "prescribed_medicine": "Sertraline", "dosage": "50mg-200mg once daily"},
    {"symptoms": "Asthma", "prescribed_medicine": "Albuterol", "dosage": "2 puffs every 4-6 hours as needed"},
    {"symptoms": "Arthritis Pain", "prescribed_medicine": "Naproxen", "dosage": "220mg-550mg twice daily with food"},
    {"symptoms": "Cholesterol Management", "prescribed_medicine": "Atorvastatin", "dosage": "10mg-80mg once daily in the evening"},
    {"symptoms": "Osteoporosis", "prescribed_medicine": "Alendronate", "dosage": "70mg once weekly"},
    {"symptoms": "Gastroesophageal Reflux", "prescribed_medicine": "Esomeprazole", "dosage": "20mg-40mg once daily before a meal"},
    {"symptoms": "Thyroid Disorder", "prescribed_medicine": "Levothyroxine", "dosage": "25mcg-300mcg once daily on an empty stomach"},
    {"symptoms": "Type 2 Diabetes", "prescribed_medicine": "Metformin", "dosage": "500mg-2000mg twice daily with meals"},
    {"symptoms": "ADHD", "prescribed_medicine": "Methylphenidate", "dosage": "5mg-20mg twice daily with or without food"},
    {"symptoms": "Hypothyroidism", "prescribed_medicine": "Synthroid", "dosage": "25mcg-300mcg once daily on an empty stomach"},
    {"symptoms": "Gout", "prescribed_medicine": "Colchicine", "dosage": "0.6mg-1.2mg at the onset of symptoms, then 0.6mg one hour later"},
    {"symptoms": "Irritable Bowel Syndrome (IBS)", "prescribed_medicine": "Dicyclomine", "dosage": "10mg-20mg three times daily before meals"},
    {"symptoms": "Opioid-induced Constipation", "prescribed_medicine": "Lubiprostone", "dosage": "24mcg twice daily with food"}
]
conversation = [
{"role": "Bot doctor", "content": "Hi, I am your doctor. How can i help you? "}
    ]
response_check = False
user_input = ""
app = Flask(__name__)
@app.route("/vitals_form", methods=['POST'])
def vitals_form():
    if request.method == "POST":
        data=request.get_json()
        patient_data = data.get('patient_data')
        user_input = data.get('user_input')
        bot_response= generate_question_with_langchain(patient_data,user_input,conversation)
        print("Bot Responce:***** ",bot_response)
        diagnosis,bot_questions = find_numbered_question_sentences(bot_response)  
        conversation.append({"role": "Patient", "content": user_input})
        
        if diagnosis!="":
            
            bot_response = diagnosis
        
        
        conversation.append({"role": "Bot doctor", "content": bot_response})
        print(type(bot_questions))
        print(type(bot_response))
        questions_modify=",".join(bot_questions)
        print(type(questions_modify))
        print(type(bot_response))
        return jsonify({'result': bot_response,'Questions':questions_modify})
    else:
        return jsonify({"No record found.Kinldy enter your vitals/symptoms"})

def Extractor(True_Condtion,False_Condtion, Sentences):
    check = False
    data =[]
    for i in Sentences:
          print(i)
          if i == True_Condtion:
               check = True
               continue

          if i == False_Condtion:

               check= False
               continue
          
          if check== True:
              data.append(i) 
    return data

#This function transformers the diagnostic responce of model into following format (Remedies:"",Medicine:"")
def transform_diagnosticResponce(bot_response_ai):
    sentences = bot_response_ai.split('\n')  
    Remedies=Extractor("Remedies: ","Medicine: ",sentences)
    Medicine =Extractor("Medicine: ","Remedies: ",sentences)
    return [Remedies,Medicine]
def find_numbered_question_sentences(bot_response_ai):
    
    # Split on newline characters and add to sentences list
    sentences = bot_response_ai.split('\n')  

    # Filter sentences that start with a number and end with a question mark
    question_sentences = []
    itr = 0
    index =[]
    for sentence in sentences:
         
         for i in sentence:
              
              if i=='?':
                   question_sentences.append(sentence)
                   index.append(itr)
                   break

         itr+=1

    if len(question_sentences)>=1:#If condition is true model has questions
         question_sentences[0]= (" ".join(sentences[:index[0]])) + question_sentences[0]
         question_sentences[-1] = question_sentences[-1] + (" ".join(sentences[index[-1]+1:]))
         
    Remedy,Medicine = transform_diagnosticResponce(bot_response_ai)
    diagnosis=""
    if len(Remedy)>0 or len(Medicine)>0:
         
         diagnosis= (" ".join(Remedy))+(" ".join(Medicine))
         
         
    return diagnosis,question_sentences

   



def get_patient_data(data):
        
        if isinstance(data, list) and len(data) > 0:
                # Assuming the first item in the list contains the column names
            columns = list(data[0].keys())
            # print("Column Names:", columns)
        else:
            print("API response does not contain valid data.")
        # columns = [col[0] for col in data]


        df = pd.DataFrame(data, columns=columns)
        new_df = pd.DataFrame(medicine_list)
        if df is not None:

            patient_data = df.groupby('phone_cell').agg({
                'sex': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'bps': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'bpd': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'sugar': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'sugar_type': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'temprature_c': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'temprature_f': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'weight_kg': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'pulse': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'oxygen': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'surgical_history': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'medical_history': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'allergies': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'medicine_history': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'diagnosis': lambda x: x.iloc[0] if not x.isnull().all() else '',
            }).reset_index()

            patient_data['content'] = df.apply(
                lambda x: f"Sex: {x['sex']}. Blood pressure systolic is {x['bps']}. Blood pressure diastolic is {x['bpd']}. Sugar is {x['sugar']}. Sugar Type is {x['sugar_type']}. Temperature (C) is {x['temprature_c']}. Temperature (F) is {x['temprature_f']}. Weight (kg) is {x['weight_kg']}. Pulse is {x['pulse']}. Oxygen is {x['oxygen']}.Patient is allergic from {x['allergies']}.Patient has history of {x['medical_history']}.Prescribed medicine is {x['medicine_history']}", 
                axis=1
            )

            patient_data = patient_data[['content', 'diagnosis']]
            #patient_data = pd.concat([patient_data, new_df], axis=1)

        else:
            patient_data = "How can we help you?"

        return patient_data



def generate_question_with_langchain(data, question, conversation):

    global response_check

    if data or question is not None:

        patient_data=get_patient_data(data)
    
    if patient_data is not None:
        try: 
            sources = []    
            for index, row in patient_data.iterrows():
                doc = Document(
                    page_content=row['content'],
                    metadata={"source": row.diagnosis},
                )
                sources.append(doc)
            
            chunks = []
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n", ".", "!", "?", ",", " ", "<br>"],
                chunk_size=300,
                chunk_overlap=0
            )
                    
            for source in sources:
                for chunk in splitter.split_text(source.page_content):
                    chunks.append(Document(page_content=chunk, metadata=source.metadata))
            # print(source)
            
            start_time = time.time()
            # print("question",question)
    
            index_object = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key='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'))
            template = """
            
            You are a doctor that is an expert in medical health and is part of a hospital system called medicare AI
            You must show and ask question about user data at first response.
            You know about symptoms mentioned in and signs of various types of illnesses.
            You have been provided with your previous chat history.
            You have been provided a medicine list to prescribe from when you have come up with a solution{medicine_list}.
            You have been provided with user's data to better understand user's ailment and to ask questions.
            You must ask questions regarding user's ailment and then give them a solution.
            You must analyze user's chat history to avoid asking same questions over and over again and to come up with a solution.
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy or medicine.
            If a response includes a need for serious medical attention with a doctor, recommend them to book an appointment with our professional healthworkers at EZSHIFA.
            You must use the following format to display the solution,(Remedies: , Medicine:).   
            Patient history: {summaries}
            User response to previous question: {question}

            """

            PROMPT = PromptTemplate(template=template, input_variables=["summaries","question","medicine_list"])

            os.environ['OPENAI_API_KEY'] = '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' 
            p_data = load_qa_with_sources_chain(OpenAIChat(openai_api_key='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', temperature=0, model_name="gpt-3.5-turbo"), prompt=PROMPT)

            if response_check == True:
                #  print("[INFO] response check true")
                 response_check = False
                 conversation[1]["content"] = question
                 del conversation[2:]

            # print("**********CONVERSATION:  ",conversation)
            ai_response = p_data({"input_documents": index_object.similarity_search(question, k=10), "question": question,"summaries":conversation,"medicine_list":medicine_list}, return_only_outputs=True)["output_text"]
        except AttributeError:
                            ai_response="How can we help you:"

    if ai_response == "How can I help? Please tell me about your symptoms.":
        response_check = True
        # print("** [INFO] ** response check value : ",response_check)
        # print(f"Elapsed Time required to process model query: {elapsed_time_2} seconds")
        return ai_response
    # print("** [INFO] ** response check value : ",response_check)
  
    return ai_response




if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app.run(host='0.0.0.0')