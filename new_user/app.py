from flask import Flask, render_template, request,jsonify
import openai
import time
# Set your OpenAI API key here
openai.api_key = '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'
user_chat={}
summary_index ={}
app = Flask(__name__)
@app.route('/chat', methods=["POST"])               
def chatbot():
    start = time.time()
    prompt ="""             
            You are an interactive AI Doctor that is an expert in medical health and is part of a hospital system called EZSHIFA.
            You know about symptoms and signs of various types of illnesses.
            You have been provided with your previous chat history with the patient.
            You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy and medicine.
            If a response includes a need for serious medical attention with a doctor, recommend them to book an appointment with our professional healthworkers at EZSHIFA.
            You can ask as many questions regarding user's ailment and then give them a solution(Remedy+Medicine).
            You must only ask one question at a time(You can never violate this).
            You must go through all possible causes regarding user's ailment before providing a solution.
            You must use the following format to display the solution,(Remedies: , Medicine:).

            """
    if request.method == "POST":
        data=request.get_json()
        user_input = data.get('user_input')
        name = data.get('name')
        num = data.get('number')
        print(user_input)
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
        check = False
        for i in user_chat[ID]:#Check if there are pending questions, if yes return them.
            if i["role"]=="user" and i["content"] is None:

                i["content"]=user_input
                check = True
                continue

            if check== True:

                print("[INFO] Pending question detected.....")
                return jsonify({'result': i["content"]})

        user_chat[ID].append({"role": "user", "content": user_input})

        for i in user_chat[ID]:
            messages.append(i)

        print("[INFO] USER CHAT: ",user_chat[ID])
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
            
            SUMMARY = GenerateSummary(user_chat[ID][summary_index[ID][0]:summary_index[ID][1]])

            del user_chat[ID][summary_index[ID][0]:summary_index[ID][1]]
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

            return jsonify({'result': question[0]})
        
        user_chat[ID].append({"role": "assistant", "content": chat_response})
        end = time.time()
        print("[INFO] Total Response Time: ",(end-start))
        return jsonify({'result': chat_response})
        


def summary(user_input):
    prompt ="""             
            You are an intelligent sentence summarizer.
            You will count the characters in input.
            You will summarize the input.
            Your should try that character count of summarized input it atleast 50 percent less than original.
            You must ensure that sumarized input has the same meaning as the original user input.
            Your response must only include the sumarized input.
            """    

    chat_history = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", # You can experiment with other models as well
    messages=[{"role": "system", "content": prompt},
              {"role": "user", "content": user_input}]
    )

    chat_response = chat_history["choices"][0]["message"]["content"]
    return chat_response

def GenerateSummary(chatbot_chat):#Summarize chat when chat becomes too long

    for i in chatbot_chat:
      
       i['content'] =  summary(i['content'])

    return chatbot_chat

def find_numbered_question_sentences(bot_response_ai):

    # Split on newline characters and add to sentences list
    sentences = bot_response_ai.split('\n')  


    # Filter sentences that end with a question mark
    question_sentences = []
    First = str()
    Last = str()
    for sentence in sentences:
                       
        if '?' in sentence:

            q = sentence.split('?')
            
            if len(q)>2:
                #q =[ i+"?" for i in q if len(i)>1]
                #question_sentences.extend(q)

                for i in range(0, len(q) - 1, 2):

                    pair = q[i]+q[i+1]
                    question_sentences.append(pair)
            
            #elif len(q)>0 and 

            else:

                question_sentences.append(sentence)
            
        else:

            if sentence==sentences[0]:
                First = sentence

            elif sentence == sentences[-1]:
                Last = sentence

    if len(question_sentences)>0:            

        print("Questions: ",question_sentences)
        question_sentences[0] =   First+ question_sentences[0]
        
        question_sentences[-1] =   question_sentences[-1]+Last   

    return question_sentences


if __name__ == "__main__":
    app.run(debug=True)