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
import logging
import random
from langchain.llms import OpenAI
api_key = "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"
model_name = "gpt-3.5-turbo"
user_conversations={}
logging.basicConfig(filename='example.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
response_check = False
user_input = ""
app = Flask(__name__)
@app.route("/vitals_form", methods=['POST'])
def vitals_form():
    if request.method == "POST":
        try:
            data = request.get_json()
            patient_data = data.get('patient_data')
            user_input = data.get('user_input')
            count = data.get('count')
            logging.info(f'Parameters: count={count}, user_input={user_input}, patien_data={patient_data}')
            if user_input == 'Delete chat[102345617289111133]':
                DeleteChat(patient_data)
            print("USER INPUT ", user_input)
            answer_suggestions = []
            bot_response, phone, cnic = generate_question_with_langchain(patient_data, user_input,count)
            logging.info(f'Function called with parameters: bot_response={bot_response}')
            bot_question = find_numbered_question_sentences(bot_response)
            # print("Bot_question")
            # print(bot_question)
            # question_sentences, non_question_sentences = modified_question_pattern(bot_response)
            
            print("count ", count)
            # Remedy,Medicine,response=remdedy_medicine_checker(bot_response)
            # if len(Remedy)>0 or len(Medicine)>0:
            #        bot_response=response
            #        print("within bot response :",bot_response)
            if bot_question is not None and len(bot_question) > 0 and int(count)>1:
                answer_suggestions = text_suggestions(bot_question)
            if count is not None:
                count=int(count)
                if count==1:
                    reponseses_list=["Hello there! How can I assist you today? ","Greetings! I'm your AI doctor, ready to lend a virtual hand. What can I do for you today?","Hi! How can I be of service to you today? ","Welcome! As your AI doctor on duty, I'm here to assist you. What can I do to make your day a bit smoother?"]
                    random_string = random.choice(reponseses_list)
                    bot_response=random_string
                if count==4:
                    conversation = process_request(cnic, phone)
                    print("TOTAL COUNT")
                    print(count)
                    bot_response_modified, return_conversation = modified_response(user_input,conversation)
                    # Remedy,Medicine,response=remdedy_medicine_checker(bot_response_modified)
                    # if Remedy is not None 
                    bot_response = bot_response_modified
                    # Remedy,Medicine,response=remdedy_medicine_checker(bot_response)
                    # if len(Remedy)>0 or len(Medicine)>0:
                    #    bot_response=response
                    print("bot_response_modified")
                    print(bot_response)
                    print("............")
                    question_sentences = []
                    answer_suggestions = []
            # print(bot_response)
            # print(answer_suggestions)
            print("Bot Response:***** ", bot_response)
            print("Suggestions *****",answer_suggestions)
            return jsonify({
                'result': bot_response,
                'Questions': [],
                'Remedy': [],
                'Medicine': [],
                'suggestion': answer_suggestions
            })
        except Exception as e:
            logging.error(f"Error processing the request: {str(e)}")
            return jsonify({"Error is ":str(e)})
    else:
        return jsonify({"error": "No record found"})


    
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
    
# def remdedy_medicine_checker(results):
   
#     extracted_response=[]

#     remedies_keywords = ["In meanwhile", "In meantime", "remedies", "remedy", "Remedies", "Remedy","tips","In the meantime","Home remedies","home remedies"]
#     medicines_keywords = ["You can try taking","recommended dosage","medications", "Medicines", "medicine", "medicines", "prescribed", "prescribed medications", "over-the-counter", "<Medicine>", "Over-the-counter"]

#     # Find the starting index of remedies and medicines
#     remedies_start_index = min((results.find(keyword) for keyword in remedies_keywords if results.find(keyword) != -1), default=-1)
#     medicines_start_index = min((results.find(keyword) for keyword in medicines_keywords if results.find(keyword) != -1), default=-1)
#     if remedies_start_index != -1 and medicines_start_index != -1:
#         # Extract remedies and medicines based on their positions
#         remedies = results[remedies_start_index:medicines_start_index].strip().splitlines()
#         medicines = results[medicines_start_index:].strip().splitlines()
#         extracted_response= results[:remedies_start_index].splitlines()
#         print("Eextracted_response",extracted_response)
#     elif medicines_start_index != -1:
#         # Extract medicines if remedies keyword is not found
#         remedies = []
#         extracted_response= results[:medicines_start_index].splitlines()
#         medicines = results[medicines_start_index:].strip().splitlines()
#         print("Eextracted_response",extracted_response)

#     else:
#         print("Keywords not found.")
#         remedies, medicines = [],[]
#     print("Remedies: ", type(remedies))
#     print("Medicines: ", type(medicines))
    
#     return remedies, medicines, extracted_response



def get_patient_data(data):
        
        if isinstance(data, list) and len(data) > 0:
                # Assuming the first item in the list contains the column names
            columns = list(data[0].keys())
            # print("Column Names:", columns)
        else:
            print("API response does not contain valid data.")
        # columns = [col[0] for col in data]
        df = pd.DataFrame(data, columns=columns)
        # new_df = pd.DataFrame(medicine_list)
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


def generate_question_with_langchain(data, question,count):
    global response_check
    if data or question is not None:
        patient_data,cnic,phone=get_patient_data(data)
        conversation=process_request(cnic,phone)
    # print("Patient_data")
    # print(patient_data)
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
            Hi! How can i help you today ?
            You MUST analyze the patient query and answer accordingly.
            You know about symptoms and signs of various types of illnesses.
            You are provided with the patient vitals and history. 
            MUST Generate one and only one follow-up question in a response as a healthcare professional responding to the patient's symptoms. 
            If a query requires serious medical attention with a doctor, recommend them to book an appointment with our doctors.
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy and must prescribed names of one or two medicines you know depend on patient symptoms in FINAL ANSWER. 
            Format any lists on individual lines with a dash and a space in front of each line.
            QUESTION: {question}
            summaries: {summaries}
            
            =========
            FINAL ANSWER:"""
    
            PROMPT = PromptTemplate(template=template, input_variables=["summaries","question"])
            os.environ['OPENAI_API_KEY'] = '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' 
            p_data = load_qa_with_sources_chain(OpenAIChat(openai_api_key='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',top_p=0.5, temperature=0.5, model_name="gpt-3.5-turbo"), prompt=PROMPT)
            
            if response_check == True:
                #  print("[INFO] response check true")
                 response_check = False
                 conversation[1]["content"] = question
                 del conversation[2:]
            # print("**********CONVERSATION:  ",conversation,"\n\n\n")
            ai_response = p_data({"input_documents": index_object.similarity_search(question, k=10), "question": question,"summaries":conversation}, return_only_outputs=True)["output_text"]
            
        except AttributeError:
                            ai_response="How can we help you?"
    else:
        ai_response="Hello! how can we help you?"

    conversation.append({"role": "Bot doctor", "content": ai_response})
    update_convo(cnic,phone,conversation)
   
    return ai_response,phone,cnic


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
    suggestion_array = []
    api_key = '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'
    openai.api_key = api_key

    for question in enumerate(bot_question):
        # print("question", question)
        prompt = f"Thoroughly analyze the given {question} and provide two to three very brief potential answers from a patient's perspective. If the {question} pertains to missing vital information, ensure the generated suggestions include highly accurate values."
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Choose the appropriate engine
            prompt=prompt,
            max_tokens=200  # Adjust as needed
        )
        suggestions = response.choices[0].text.strip().splitlines()
        for s in suggestions:
            removed_colon = s.find(":")
            if removed_colon != -1:
                result = str(s[removed_colon + 1:].strip())
                suggestion_array.append(result)
            else:
                suggestion_array.append(str(s))

    print("suggestions")
    print(suggestion_array)
    return suggestion_array


def modified_response(user_input,conversation):
    #    questions_asked=find_numbered_question_sentences(bot_response)
    #     print(conversation)
    # if len(questions_asked)>=0:
        prompt = f"""
        AI DOCTOR asked questions: {conversation}
        PATIENT responses: {user_input}
    """
        context_prompt = """
    You are interacting with a Healthcare chatbot of EZshifa designed to assist with general 
    health inquiries and provide guidance based on reported symptoms and concerns.Generate brief response including information about home remedies and prescribed one or two medicines from the provided medicine list for the following health symptoms.]
    """    
        # memory = []
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                  {"role": "Bot doctor", "content": context_prompt},
                  {"role": "user", "content": prompt},
                ],
                max_tokens=500
            )
        # memory.append(response.choices[0]['message']['content'])
        bot_response=response.choices[0]['message']['content']
        conversation.append({"role": "Bot doctor", "content": bot_response})
        return bot_response,conversation
        
            
if __name__ == "__main__":
     app.run(host='127.0.0.1', port=8080, debug=True)
    #   app.run(host='0.0.0.0', port=3012, debug=True)