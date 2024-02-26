import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.vectorstores.faiss import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request,jsonify,g
import time
import os
import openai
from langchain.llms import OpenAI





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
count=0
user_input = ""
app = Flask(__name__)
@app.route("/vitals_form", methods=['POST'])
def vitals_form():
    if request.method == "POST":
        data=request.get_json()
        patient_data = data.get('patient_data')
        user_input = data.get('user_input')
        if user_input=='Delete chat[102345617289111133]':   
             DeleteChat(patient_data)
        print("USER INPUT ############",user_input)
        answer_suggestions=[]
        bot_response,count= generate_question_with_langchain(patient_data,user_input)
        bot_question = find_numbered_question_sentences(bot_response)
        Remedy,Medicine,Recommend = transform_diagnosticResponce(bot_response)
        Recommend =check_data(Remedy,Medicine,Recommend,bot_question)
        question_sentences,non_question_Sentences=modified_question_pattern(bot_response)
        print("Bot Responce:***** ",bot_response) 
        # print("Questions :",question_sentences)
        # if len(question_sentences)==len(user_input):
        print("count ",count)
        if count>=3:
                bot_response_modified=modified_response(question_sentences,user_input,bot_response,user_conversations)
                Remedy,Medicine,Recommend = transform_diagnosticResponce(bot_response)
                Recommend =check_data(Remedy,Medicine,Recommend,bot_question)
                non_question_Sentences=bot_response_modified
        if len(Remedy)>=1 or len(Medicine)>=1:
             bot_response = Recommend
             bot_question = []
        if bot_question is not None and len(bot_question)>0:
             answer_suggestions=text_suggestions(bot_question)
        # print("****Suggestions: ****",answer_suggestions)
        return jsonify({'result':non_question_Sentences,'Questions': question_sentences,'Remedy':Remedy,'Medicine':Medicine,'suggestion':answer_suggestions})
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
          
          if i[:len(True_Condtion)] == True_Condtion :
               check = True
               
          if i[:len(False_Condtion)] == False_Condtion :

               check= False
               
          
          if check== True:
              data.append(i)
      
    return data


def find_sentences_with_word(sentences, target_word,stop_word):
    result_list = []
    check = False
    for sentence in sentences:
        lower_sentence = sentence.lower()  # Convert to lowercase
        
        if target_word[0] in lower_sentence or target_word[1] in lower_sentence:
            check = True
        if stop_word!=None:
          if stop_word[0] in lower_sentence or stop_word[1] in lower_sentence:
            break
        if check:
            result_list.append(sentence)
     

    return result_list

# def check_data(remedy,medicine,recommend,questions):

#     check = False
#     if remedy is not None and medicine is not None and recommend is not None:
#         for i in questions:
#             if i in remedy or i in medicine or i in recommend:
#                 check = True
#                 break
#         if check == True:
#             del remedy[:]
#             del medicine[:]
#             recommend=''
#     return recommend



def check_data(remedy, medicine, recommend, questions):
    check = False
    if remedy is not None and medicine is not None and recommend is not None:
        if questions is not None:
            for i in questions:
                if i in remedy or i in medicine or i in recommend:
                    check = True
                    break

            if check:
                del remedy[:]
                del medicine[:]
                recommend = ''

    return recommend



#This function transformers the diagnostic responce of model into following format (Remedies:"",Medicine:"")
def transform_diagnosticResponce(bot_response_ai):
    # Split on newline characters and add to sentences list
    sentences = bot_response_ai.split('\n')  
    Remedies=Extractor("Remedies: ","Medicine: ",sentences)
    Medicine =Extractor("Medicine: ","Remedies: ",sentences)
    recommend = str()
    if len(Remedies)==0 and len(Medicine)==0: 
     Remedies=find_sentences_with_word(sentences,["Remedies","Remedy","Tips","Tip","In the meantime"],None)
     Medicine =find_sentences_with_word(sentences,["Medicine","Medicines"],None)
    #  recommend = str()
    for i in sentences:
          if i not in Remedies and i not in Medicine:
              recommend +=i


    return [Remedies,Medicine,recommend]

def find_numbered_question_sentences(bot_response_ai):
    used_phrases = set()

    # Split on newline characters and add to sentences list
    sentences = bot_response_ai.split('\n')  
    # print(sentences)
    # Filter sentences that end with a question mark
    question_sentences = []
    itr = 0
    index =[]
    for sentence in sentences:
         for i in sentence:
              
              if i=='?':
                   sen=''.join([i for i in sentence if not i.isdigit()])
                   question_sentences.append(sen)
                   index.append(itr)
                   break
         itr+=1
    if len(question_sentences)>=1:
           question_sentences[0]= (" ".join(sentences[:index[0]])) + question_sentences[0]
           return question_sentences
       
       
def modified_question_pattern(bot_response_ai):
    # Initialize empty lists to store question sentences and non-question sentences
        sentences = bot_response_ai.split('\n')  
        # print(sentences)
        question_sentences = []
        non_question_sentences = []
        index = []

        # Initialize iterator variable
        itr = 0

        # Iterate through each sentence in the 'sentences' list
        for sentence in sentences:
            # Print the current sentence
            # print(sentence)

            # Check if the sentence ends with a question mark
            if sentence.endswith('?'):
                # Remove digits from the sentence
                sen = ''.join([char for char in sentence if not char.isdigit()])
                # print("******Questions ending with ?******")
                # print(sen)

                # Append the modified sentence to 'question_sentences'
                question_sentences.append(sen)

                # Append the index of the sentence to 'index'
                index.append(itr)
            else:
                # If the sentence does not end with a question mark, save it to 'non_question_sentences'
                # print("$$$$ without ?$$")
                # print(sentences)
                non_question_sentences.append(sentence)

            # Increment the iterator
            itr += 1

        # # Print the results
        # print("Question Sentences:")
        # for q_sentence in question_sentences:
        #     print(q_sentence)

        # print("\nNon-Question Sentences:")
        # for non_q_sentence in non_question_sentences:
            
        return question_sentences,non_question_sentences
    
    

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
                lambda x: f"Sex: {x['sex']}.Blood pressure systolic is {x['bps']}. Blood pressure diastolic is {x['bpd']}.Sugar is {x['sugar']}.Temperature is {x['temprature_c']}. Temperature is {x['temprature_f']}. Weight is {x['weight_kg']}. Pulse is {x['pulse']}. Oxygen is {x['oxygen']}.Patient is allergic from {x['allergies']}.Patient has history of {x['medical_history']}.Prescribed medicine is {x['medicine_history']}", 
                axis=1
            )

            patient_data = patient_data[['content', 'diagnosis']]
            #patient_data = pd.concat([patient_data, new_df], axis=1)

        else:
            patient_data = "How can we help you?"

        return patient_data,cnic,phone


# def chat_templates():


def generate_question_with_langchain(data, question):
    global response_check
    global count
    if data or question is not None:
        patient_data,cnic,phone=get_patient_data(data)
        conversation=process_request(cnic,phone)
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
                chunk_size=200,
                chunk_overlap=0
            )
            
            for source in sources:
                for chunk in splitter.split_text(source.page_content):
                    chunks.append(Document(page_content=chunk, metadata=source.metadata))
            start_time = time.time()
            index_object = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key='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'))
            template = """
            You are an AI Doctor that is an expert in medical health and is part of a hospital system called EZSHIFA.
            You MUST analyze the patient query and answer accordingly. DO NOT ask irrelvent questions.
            You know about symptoms and signs of various types of illnesses.
            You are provided with the patient vitals and history. 
            Generate one and only one follow-up question in a response as a healthcare professional responding to the patient's initial information.
            YOU MUST PROVIDE THE SOLUTION AFTER ONE OR TWO QUESTIONS.
            Ask about specific details, symptoms, or any relevant information to further assess the patient's condition and provide appropriate guidance or recommendations.
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy.But DO NOT provide solution while asking questions.
            If a query requires serious medical attention with a doctor, recommend them to book an appointment with our doctors
            YOU MUST follow the format while providing the FINAL ANSWER after analyzing the patient response (Remedies/Remedy/Tips:, Medicines/Dozage {medicine_list}:)
            Format any lists on individual lines with a dash and a space in front of each line.
            QUESTION: {question}
            =========
            {summaries}
            =========
            FINAL ANSWER:"""
    
            PROMPT = PromptTemplate(template=template, input_variables=["summaries","question","medicine_list"])
            os.environ['OPENAI_API_KEY'] = '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' 
            p_data = load_qa_with_sources_chain(OpenAIChat(openai_api_key='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',top_p=0.5, temperature=0.5, model_name="gpt-3.5-turbo"), prompt=PROMPT)
            
            if response_check == True:
                #  print("[INFO] response check true")
                 response_check = False
                 conversation[1]["content"] = question
                 del conversation[2:]
            # print("**********CONVERSATION:  ",conversation,"\n\n\n")
            ai_response = p_data({"input_documents": index_object.similarity_search(question, k=10), "question": question,"summaries":conversation,"medicine_list":medicine_list}, return_only_outputs=True)["output_text"]
            
            
        except AttributeError:
                            ai_response="How can we help you?"

    conversation.append({"role": "Bot doctor", "content": ai_response})
    count=count+1
    update_convo(cnic,phone,conversation)
    # print("** [INFO] ** response check value : ",response_check)
    end_time = time.time()
    elapsed_time_2 = end_time - start_time
    # print(f"Elapsed Time required to process model query: {elapsed_time_2} seconds")
    return ai_response,count


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

def DeleteChat(data):
     
     patient_data,cnic,phone=get_patient_data(data)
     user_id = str(cnic)+str(phone)
     del user_conversations[user_id]

def text_suggestions(bot_question):
    suggestion_array=[]
#    bot_question= [
#         " 1. Can you please provide your age?",
#         "2. Are you currently experiencing any symptoms or discomfort?",
#         "3. Have you recently been diagnosed with any medical conditions or undergone any medical procedures?",
#         "4. Are you currently taking any medications other than disprin and panadol?",
#         "5. Have you experienced any recent changes in your lifestyle or diet?"
#     ]
    # user_input=["28","no","no","no","no"]
    # for i, question in enumerate(bot_question):
    #     context = f"{question} {user_input[i]}"
    #     print(question)
    api_key='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'
    openai.api_key = api_key
    for question in enumerate(bot_question):
        #    print("question",question)
           prompt = f"Analyze the {question} carefully. Generate three possible answers of question as a PATIENT.Generated answers MUST be complete and to the point."
           response = openai.Completion.create( 
                    model="gpt-3.5-turbo-instruct", # Choose the appropriate engine
                    prompt=prompt,
                    max_tokens=50 # Adjust as needed
                )
           suggestion = response.choices[0].text.strip()
           suggestion_array.append(suggestion)
    return suggestion_array


# def check_length(questions,user_input):
#         # Convert both questions and answers to lowercase for case-insensitive comparison
#     lower_questions = [question.lower() for question in questions]
#     lower_answers = [user_input.lower() for user_input in user_input]
#     # Check if each question is in the answers
    
#     if len(lower_questions)==len(lower_answers):
#          flag=True
#     else:
#         flag=False
#     return flag
    

def modified_response(question_sentences,user_input,bot_response,conversation):
    #    questions_asked=find_numbered_question_sentences(bot_response)
    #     print(conversation)
    # if len(questions_asked)>=0:
        prompt = f"""
        AI DOCTOR asked questions: {question_sentences}
        PATIENT responses: {user_input}
    """
        context_prompt = """
    You are interacting with a Healthcare chatbot designed to assist with general 
    health inquiries and provide guidance based on reported symptoms and concerns.Generate a detailed response including information about Remedies and any suggested Medicines {medicine_list} for the following health condition:[{user_input}]
    """    
        # memory = []
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                  {"role": "system", "content": context_prompt},
                  {"role": "user", "content": prompt},
                ],
                max_tokens=500
            )
        # memory.append(response.choices[0]['message']['content'])
        bot_response=response.choices[0]['message']['content']
    
    # else:
    #     bot_response=bot_response
    # conversation.append({"role": "Bot doctor", "content": bot_response})
        return bot_response
        
            
if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app.run(host='0.0.0.0')