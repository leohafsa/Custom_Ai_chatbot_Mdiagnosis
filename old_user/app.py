from flask import Flask, render_template, request,jsonify
import openai
import time
from util import *
import logging



# Set your OpenAI API key here
openai.api_key = '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'
user_chat={}
summary_index ={}
logging.basicConfig(filename='olduser.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
@app.route('/chatbot', methods=["POST"])               
def chatbot():
    start = time.time()
    data=request.get_json()
    patient_data = data.get('patient_data')
    user_input = data.get('user_input')
    relevant_data,name,num=get_patient_data(patient_data)

    #print(patient_data.columns)
    
    #relevant_data = getMedicalData(user_input,openai,patient_data)
    

    print("[INFO] Patient Summary: ",relevant_data)
#Patient Medical Data:.
#You have been provided with patient data(Medical History/Medical Diagnosis/Vital Information) relevant to the query, you must only use it when necessary. 
    prompt =f"""   
            You are an interactive AI Doctor that is an expert in medical health and is part of a hospital system called EZSHIFA.
            You know about symptoms and signs of various types of illnesses.
            You have been provided with your previous chat history with the patient.
            You have been provided with EZSHIFA patient past records.
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy and medicine.
            If a response includes a need for serious medical attention with a doctor, recommend them to book an appointment with our professional healthworkers at EZSHIFA.
            You must ask as many questions as you like regarding user's ailment and then give them a solution(Remedy+Medicine).
            You must only ask one question at a time(You can never violate this).
            You must go through all possible causes regarding user's ailment before providing a solution.
            You must use the following format to display the solution,(Remedies: , Medicine:).
            {relevant_data}
            """             
    
    print("PROMPT: ",prompt)    
    if name == None:

        name = generate_random_string()

    if num == None:
        num = generate_random_string()

    #print(user_input)

    ID = str(name)+str(num)

    if  user_input =='Delete chat[102345617289111133]':
        print("Del")    
        del user_chat[ID]
        del summary_index[ID]


    else:
        

        if ID not in user_chat.keys():
            print("New ID")
            user_chat[ID] = []
            summary_index[ID] = None
            user_input="Hello"
        
        messages=[{"role": "system", "content": prompt}]
        
        #messages=[{"role": "assistant", "content": f'Patient Records:{relevant_data}'}]
        check = False
        for i in user_chat[ID]:#Check if there are pending questions, if yes return them.

            
            if i["role"]=="user" and i["content"] is None:

                i["content"]=user_input
                check = True
                continue

            if check== True:

                print("[INFO] Pending question detected.....")
                suggestions = text_suggestions(i['content'],openai)
                
                logging.info(f'result={i["content"]}, user_input={user_input}, num={num},name={name}')
                return jsonify({'result': i["content"],'suggestion': suggestions,'Questions': [],'Remedy': [],'Medicine': []})







        user_chat[ID].append({"role": "user", "content": user_input})

        for i in user_chat[ID]:
            messages.append(i)


        print("[INFO] USER CHAT: ",user_chat[ID])
        #print("[INFO] MESSAGES: ",messages)

        chat_history = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # You can experiment with other models as well
        messages=messages,
        )

        chat_response = chat_history["choices"][0]["message"]["content"]

        token_count = chat_history['usage']['total_tokens']
        output_token = len(chat_response)
        input_token = token_count - output_token

        print("[INFO] GPT RESPONSE: ",chat_response)

        #print('Total tokens used:',token_count)
        #print("Input token: ",input_token)

        
        if token_count>3000:# If chat becomes too long summarize half of it to reduce tokens and keep context.
    
            print("SUMMARIZING....")
        
            if summary_index[ID] is None:
                
                summary_index[ID]=[2,len(user_chat[ID])//2]

            else:
                
                summary_index[ID][1] = len(user_chat[ID])//2
            
            SUMMARY = GenerateSummary(user_chat[ID][summary_index[ID][0]:summary_index[ID][1]],openai)

            #del user_chat[ID][summary_index[ID][0]:summary_index[ID][1]]
            user_chat[ID][summary_index[ID][0]:summary_index[ID][1]] = SUMMARY

            #summary_index[ID][0]=summary_index[ID][1]
            print(user_chat[ID])

        question = find_numbered_question_sentences(chat_response)

        
        if len(question)>2:#If bots response is more than question threshold, extract questions and them to chat.

            print("[INFO] Questions length violation detected.....")
            for i in question:
                
                if len(i)>0:

                    user_chat[ID].append({"role": "assistant", "content": i})
                    user_chat[ID].append({"role": "user", "content": None})

            suggestions = text_suggestions(question[0],openai)
            logging.info(f'result={question[0]}, user_input={user_input}, num={num},name={name}')
            return jsonify({'result': question[0],'suggestion': suggestions,'Questions': [],'Remedy': [],'Medicine': []})
    
        if len(question)>0:

            question = "".join(question)
            suggestions = text_suggestions(question,openai)
            user_chat[ID].append({"role": "assistant", "content": question})
            logging.info(f'result={question}, user_input={user_input}, num={num},name={name}')
            return jsonify({'result': question,'suggestion': suggestions,'Questions': [],'Remedy': [],'Medicine': []})

        
        user_chat[ID].append({"role": "assistant", "content": chat_response})

        end = time.time()
        print("[INFO] Total Response Time: ",(end-start))
        logging.info(f'result={chat_response}, user_input={user_input}, num={num},name={name}')
        return jsonify({'result': chat_response,'suggestion': None,'Questions': [],'Remedy': [],'Medicine': []})
        #return chat_response













# @app.route('/getchat', methods=["GET"]) 
def returnchat():
    data=request.get_json()
    user_input = data.get('user_input')
    name = data.get('name')
    num = data.get('number')
    print(user_input)
    ID = str(name)+str(num)
    return jsonify({'result': user_chat[ID],'name':name,'num':num})
        #return chat_response





if __name__ == "__main__":
    app.run(host='127.0.0.1',debug=True, port=8080)