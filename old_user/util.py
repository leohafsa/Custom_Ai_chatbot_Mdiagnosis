import random
import string
import pandas as pd

import pandas as pd

def find_min_max(df, column_name):
    try:
        # Convert the specified column to integers
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce', downcast='integer')

        # Drop rows with NaN values after conversion
        df = df.dropna(subset=[column_name])

        # Find the min and max values
        min_value = df[column_name].min()
        max_value = df[column_name].max()

        return min_value, max_value

    except KeyError:
        return f"Column '{column_name}' not found in the DataFrame."
    except ValueError:
        return f"Unable to convert values in '{column_name}' to integers."


def generate_random_string(length=5):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def text_suggestions(question,openai):
    suggestion_array = []
    #api_key = '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'
    #openai.api_key = api_key

    
    
    
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

# print("suggestions")
    # print(suggestion_array)
    return suggestion_array

def summary(user_input,openai):

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



def getMedicalData(user_input,openai,patient_Data):

    prompt ="""             
            You will be provided with patient data.You should generate a short summary of patient data.
            The summary should be as short as possible and it should have the important information from original data.
            Below is  sample response:
            "Female patient data includes blood pressure ranging from 120/70 to 134/89, sugar levels range from 78 to 219, temperature varies from 95 to 100.3, pulse ranges from 75 to 95, oxygen levels are consistently around 99-100. No allergies, medical history, prescribed medicine, or diagnosis mentioned in the records."
            You response must be strictly in above format.
            """    

    user_input = f'Patient data is:{patient_Data}.'
    chat_history = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", # You can experiment with other models as well
    messages=[{"role": "system", "content": prompt},
              {"role": "user", "content": user_input}]
    )

    chat_response = chat_history["choices"][0]["message"]["content"]




    return chat_response



def GenerateSummary(chatbot_chat,openai):#Summarize chat when chat becomes too long

    for i in chatbot_chat:
      
       i['content'] =  summary(i['content'],openai)

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

                    pair = q[i]+"?"+q[i+1]+"?"
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


def get_patient_data(data):
        
        if isinstance(data, list) and len(data) > 0:
                # Assuming the first item in the list contains the column names
            columns = list(data[0].keys())
            # print("Column Names:", columns)
        else:
            print("API response does not contain valid data.")
        # columns = [col[0] for col in data]
        df = pd.DataFrame(data, columns=columns)
        #df= df.to_string(index=False)
        # new_df = pd.DataFrame(medicine_list)

        #print(df.head())
        
        cnic =str(df['cnic'][0])
        phone =str(df['phone_cell'][0])

        col = {'bps':"Blood pressure systolic", 'bpd':" Blood pressure diastolic" ,
               'medicine_history':"Prescribed medicine",'diagnosis':"Patient diagnosis", }
        Data=[]
        itr = 0
        """for i in df.iterrows():

                

                row ="Sex: "+i['sex']+"."+"Blood pressure systolic is "+ i['bps']+"."+"Blood pressure diastolic is "+i['bpd']+"."+"Sugar is "+i['sugar']+"."+"Temperature is "+i['temprature_c']+"."+"Temperature is "+i['temprature_f']+"."+"Weight is "+i['weight_kg']+"."+"Pulse is "+i['pulse']+"."+"Oxygen is "+i['oxygen']+"."+"Patient is allergic from" +i['allergies']+"."+"Patient has history of "+i['medical_history']+"."+"Prescribed medicine is "+i['medicine_history']+"."+"Patient diagnosis is "+i['diagnosis']+"."

                Data.append(row)
        """

        BPS_max,BPS_min = find_min_max(df,'bps')
        BPD_max,BPD_min = find_min_max(df,'bpd')
        sugar_max,sugar_min = find_min_max(df,'sugar')
        Temp_max,Temp_min = find_min_max(df,'temprature_f')
        Weight_max,Weight_min = find_min_max(df,'weight_kg')
        Pulse_max,Pulse_min = find_min_max(df,'pulse')
        Oxy_max,Oxy_min = find_min_max(df,'oxygen')

        sur_string = "Surgical History:"+', '.join(df['surgical_history'].astype(str).unique())
        all_string = "Allergy History:" +', '.join(df['allergies'].astype(str).unique())
        med_names_string = "Medicine History:"+', '.join(df['medicine_history'].astype(str).unique())
        
        medical_names_string = "Medical History:"+', '.join(df['medical_history'].astype(str).unique())
        diag_names_string = "Diagnosis History:"+', '.join(df['diagnosis'].astype(str).unique())
         
        # Assuming you have already calculated the min and max values
        # for each specified column using find_min_max function

        # Construct strings in the relevant format
        blood_sys_string = f"Sytolic: Ranging from {BPS_min} to {BPS_max}"
        blood_dys_string = f"Dystolic: Ranging from {BPD_min} to {BPD_max}"
        blood_pressure_string = blood_sys_string +","+ blood_dys_string
        Weight_string = f"Oxygen levels: Varies from {Weight_min} to {Weight_max}"
        sugar_levels_string = f"Sugar levels: Ranging from {sugar_min} to {sugar_max}"
        temperature_string = f"Temperature: Varies from {Temp_min} to {Temp_max}"
        pulse_string = f"Pulse: Ranges from {Pulse_min} to {Pulse_max}"
        oxygen_levels_string = f"Oxygen levels: Varies from {Oxy_min} to {Oxy_max}"
       
        # Combine all strings into a single string
        result_string ="Patient Past Records:" +"\n".join([blood_pressure_string, sugar_levels_string, temperature_string, pulse_string, oxygen_levels_string,Weight_string,med_names_string,medical_names_string,diag_names_string,sur_string,all_string])

        # Print or use the result_string as needed
        print(result_string)

        
        return result_string,cnic,phone
