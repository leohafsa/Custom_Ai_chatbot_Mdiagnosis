import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.vectorstores.faiss import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request,jsonify,g
import openai
import time
import os
import requests
import re




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


user_conversations={}
response_check = False
user_input = ""
app = Flask(__name__)





@app.route("/vitals_form", methods=['POST'])

def vitals_form():


    if request.method == "POST":
        data=request.get_json()
        patient_data = data.get('patient_data')
        user_input = data.get('user_input')
        #conversation.append({"role": "Patient", "content": user_input})
        bot_response= generate_question_with_langchain(patient_data,user_input)
        # speech_bot=text_to_speech(bot_response)
        # print(bot_response)
        print("Bot Responce:***** ",bot_response)
        #bot_response = find_question_sentences(bot_response)
        Remedy,Medicine,Recommend = transform_diagnosticResponce(bot_response)
        #conversation.append({"role": "Bot doctor", "content": bot_response})
        if len(Remedy)>=1 or len(Medicine)>=1:
             bot_response = Recommend
             bot_question = []
        else:
             bot_question = find_numbered_question_sentences(bot_response)
        return jsonify({'result': bot_response,'Questions': bot_question,'Remedy':Remedy,'Medicine':Medicine})
    else:
        return jsonify({"No record found"})


def Extractor(True_Condtion,False_Condtion, Sentences):
     
    check = False
    data =[]
    for i in Sentences:
          
          if i == True_Condtion:
               check = True
               continue

          if i == False_Condtion:

               check= False
               continue
          
          if check== True:
              data.append(i)

    if len(data)>=1:
         return data

    for i in Sentences:
          
          if i[:len(True_Condtion)] == True_Condtion or i[:len(True_Condtion)-1] == True_Condtion:
               check = True
               
          if i[:len(False_Condtion)] == False_Condtion or i[:len(False_Condtion)-1] == False_Condtion :

               check= False
               
          
          if check== True:
              data.append(i)
           
    return data



def find_sentences_with_word(sentences, target_word,stop_word):
    result_list = []
    check = False
    for sentence in sentences:
        if target_word in sentence:
            #result_list.append(sentence)
            check = True
        if stop_word in sentence:
               break
        if check == True:
             result_list.append(sentence)
    return result_list
#This function transformers the diagnostic responce of model into following format (Remedies:"",Medicine:"")
def transform_diagnosticResponce(bot_response_ai):
    sentences = bot_response_ai.split('\n')  
    Remedies=Extractor("Remedies: ","Medicine: ",sentences)
    Medicine =Extractor("Medicine: ","Remedies: ",sentences)
    Remedies=find_sentences_with_word(sentences,"remedies","medicine")
    Medicine =find_sentences_with_word(sentences,"medicine","KKK")
    recommend = str()
    for i in sentences:
         if i not in Remedies and i not in Medicine:
              recommend +=i

    return [Remedies,Medicine,recommend]

def find_numbered_question_sentences(bot_response_ai):

    # Split on newline characters and add to sentences list
    sentences = bot_response_ai.split('\n')  


    # Filter sentences that end with a question mark
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
 
    print("question_sentences", question_sentences)
    return question_sentences

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
        cnic ="1"
        phone ="1"

        if df is not None:

            patient_data = df.groupby('phone_cell').agg({
                'cnic': lambda x: x.iloc[0] if not x.isnull().all() else '',
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

            cnic = patient_data['cnic']
            phone = patient_data['phone_cell']
            patient_data['content'] = df.apply(
                lambda x: f"Sex: {x['sex']}. CNIC is {x['cnic']}. Blood pressure systolic is {x['bps']}. Blood pressure diastolic is {x['bpd']}. Sugar is {x['sugar']}. Sugar Type is {x['sugar_type']}. Temperature (C) is {x['temprature_c']}. Temperature (F) is {x['temprature_f']}. Weight (kg) is {x['weight_kg']}. Pulse is {x['pulse']}. Oxygen is {x['oxygen']}.Patient is allergic from {x['allergies']}.Patient has history of {x['medical_history']}.Prescribed medicine is {x['medicine_history']}", 
                axis=1
            )

            patient_data = patient_data[['content', 'diagnosis']]
            #patient_data = pd.concat([patient_data, new_df], axis=1)

        else:
            patient_data = "How can we help you?"

        return patient_data,cnic,phone



def generate_question_with_langchain(data, question):

    global response_check

    if data or question is not None:

        patient_data,cnic,phone=get_patient_data(data)

        conversation=process_request(cnic,phone)

        conversation.append({"role": "Patient", "content": question})
    
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
            start_time = time.time()
    
            index_object = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key='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'))
            #print("question",question)
            
            #Doctor will ask questions about the patient's vitals and conditions at first response.Doctor will ask about symptoms.
            template = """
            
            You are an AI Doctor that is an expert in medical health and is part of a hospital system called EZSHIFA.
            You know about symptoms and signs of various types of illnesses.
            You have been provided with your previous chat history.
            You have been provided a medicine list to prescribe from when you have come up with a solution{medicine_list}.
            You have been provided with user's previous medical data.
            You must analyze user's chat history to avoid asking same questions over and over again and to come up with a solution.
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy and medicine.
            If a response includes a need for serious medical attention with a doctor, recommend them to book an appointment with our professional healthworkers at EZSHIFA.
            You must ask questions regarding user's ailment and then give them a solution.
            You must ask atleast 5 questions regarding user's ailment and then provided a solution.
            You must use the following format to display the solution,(Remedies: , Medicine:).
                        
            Chat history: {summaries}
            User response to previous question: {question}

            """

            PROMPT = PromptTemplate(template=template, input_variables=["summaries","question","medicine_list"])
            

            os.environ['OPENAI_API_KEY'] = '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' 
            p_data = load_qa_with_sources_chain(OpenAIChat(openai_api_key='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', temperature=0, model_name="gpt-3.5-turbo"), prompt=PROMPT)

            if response_check == True:
                 print("[INFO] response check true")
                 response_check = False
                 conversation[1]["content"] = question

                 del conversation[2:]

            print("**********CONVERSATION:  ",conversation,"\n\n\n")
            ai_response = p_data({"input_documents": index_object.similarity_search(question, k=10), "question": question,"summaries":conversation,"medicine_list":medicine_list}, return_only_outputs=True)["output_text"]
            
            
        except AttributeError:
                            ai_response="How can we help you:"

    conversation.append({"role": "Bot doctor", "content": ai_response})

    update_convo(cnic,phone,conversation)
    print("** [INFO] ** response check value : ",response_check)
    end_time = time.time()
    elapsed_time_2 = end_time - start_time
    print(f"Elapsed Time required to process model query: {elapsed_time_2} seconds")
    return ai_response


def process_request(cnic,phone):
    # Get the user's identifier (you may want to use a user ID from authentication)
    user_id = str(cnic)+str(phone)

    # Check if the user has an existing conversation
    if user_id in user_conversations:
        conversation = user_conversations[user_id]
    else:
        # If not, initialize a new conversation
        user_conversations[user_id]=conversation = [
                           {"role": "Bot doctor", "content": "Hi, I am your AI doctor. How can I help you?"}
                                                    ]

  
    return user_conversations[user_id]

def update_convo(cnic,phone,convo):
     
     user_id = str(cnic)+str(phone)
     del user_conversations[user_id]

     user_conversations[user_id] = convo




if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app.run(host='0.0.0.0')