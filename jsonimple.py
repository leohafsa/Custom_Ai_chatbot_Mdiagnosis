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
            print("count ", count)
            if user_input == 'Delete chat[102345617289111133]':
                DeleteChat(patient_data)
            print("USER INPUT ", user_input)
            answer_suggestions = []
            bot_response, phone, cnic = generate_question_with_langchain(patient_data, user_input,count)  
            bot_question = find_numbered_question_sentences(bot_response)
            count = int(count) if count is not None else 0
            if bot_question is not None and len(bot_question) > 0 and count>1:
                answer_suggestions = text_suggestions(bot_question)
            if count<=1:
                reponseses_list=["Hello there! How can I assist you today? ","Greetings! I'm your AI doctor, ready to lend a virtual hand. What can I do for you today?","Hi! How can I be of service to you today? ","Welcome! How can I assist you today?"]
                random_string = random.choice(reponseses_list)
                bot_response=random_string
                answer_suggestions = []
            if count==4:
                conversation = process_request(cnic, phone)
                print("TOTAL COUNT")
                print(count)
                bot_response_modified, return_conversation = modified_response(user_input,conversation)
                bot_response = bot_response_modified
                print("bot_response_modified")
                print(bot_response)
                print("............")
                question_sentences = []
                answer_suggestions = []
            logging.info(f'Parameters: count={count}, user_input={user_input}, patien_data={patient_data}')
            logging.info(f'BOT RESPONSES and answer suggestions: bot_response={bot_response},answer_suggestions={answer_suggestions}')
            print("Bot Response:***** ",bot_response)
            print("Suggestion:***** ",answer_suggestions)
            return jsonify({
                'result': bot_response,
                'Questions': [],
                'Remedy': [],
                'Medicine': [],
                'suggestion': answer_suggestions
            })
        except Exception as e:
            logging.error(f"Error processing the request:{str(e)}")
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
            You are an interactive AI Doctor that is an expert in medical health and is part of a hospital system called EZSHIFA.You have the previous patient vitals and history.
            You must ask one and only one follow up question at a time(You can never violate this) as AI doctor.You can ask as many questions regarding user's ailment and then give them a solution(Remedy+Medicine).
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy and medicine.
            If a response includes a need for serious medical attention with a doctor, recommend them to book an appointment with our professional healthworkers at EZSHIFA.
            You have been provided with your previous chat history with the patient to maintain the context.
            You must go through all possible causes regarding user's ailment before providing a solution.
            You must use the following format to display the solution,(Remedies: , Medicine:).
            {question}
            {summaries}"""
    
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
        prompt = f"Thoroughly analyze the given {question} and provide two to three one line answers from a patient's perspective. If the {question} pertains to missing vital information, ensure the generated suggestions include highly accurate values."
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

    # print("suggestions")
    # print(suggestion_array)
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
    health inquiries and provide guidance based on reported symptoms and concerns.Generate brief response including information about home remedies and prescribed one or two medicines from the provided medicine list for the following health symptoms.
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
        conversation.append({"role": "Bot doctor", "content": bot_response})
        return bot_response,conversation
        
            
if __name__ == "__main__":
     app.run(host='127.0.0.1', port=8080)
    #   app.run(host='0.0.0.0', port=3012, debug=True)