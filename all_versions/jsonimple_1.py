import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.vectorstores.faiss import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request,jsonify
import re
import openai
import time
api_key = "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"
model_name = "gpt-3.5-turbo"
############################# Additional information ##################################
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
###########################################################################################################
user_input = ""
app = Flask(__name__)
@app.route("/vitals_form", methods=['POST'])
def vitals_form():
    conversation = [
    {"role": "Bot doctor", "content": " "},
    {"role": "Patient", "content": " "},
      ]
    if request.method == "POST":
        data=request.get_json()
        patient_data = data.get('patient_data')
        user_input = data.get('user_input')
        bot_response,question_array= generate_question_with_langchain(patient_data,user_input,conversation)
        # speech_bot=text_to_speech(bot_response)
        conversation.append({"role": "Patient", "content": user_input})
        conversation.append({"role": "Bot doctor", "content": bot_response})
        return jsonify({'result': bot_response,'Questions':question_array})
    else:
        return jsonify({"No record found"})

def generate_question_with_langchain(data,question,conversation):
  
    if data or question is not None:
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
                'cnic': lambda x: x.iloc[0] if not x.isnull().all() else '',
                # 'language': lambda x: x.iloc[0] if not x.isnull().all() else '',
                # 'fname': lambda x: x.iloc[0] if not x.isnull().all() else '',
                # 'lname': lambda x: x.iloc[0] if not x.isnull().all() else '',
                # 'DOB': lambda x: x.iloc[0] if not x.isnull().all() else '',
                # 'city': lambda x: x.iloc[0] if not x.isnull().all() else '',
                'sex': lambda x: x.iloc[0] if not x.isnull().all() else '',
                # 'appt_id': lambda x: x.iloc[0] if not x.isnull().all() else '',
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
                # lambda x: f"Patient phone is {x['phone_cell']}. CNIC is {x['cnic']}. Language is {x['language']}. First name: {x['fname']}. Last name is {x['lname']}. Date of Birth is  {x['DOB']}. City is  {x['city']}. Sex: {x['sex']}. Appointment ID is {x['appt_id']}. Blood pressure systolic is {x['bps']}. Blood pressure diastolic is {x['bpd']}. Sugar is {x['sugar']}. Sugar Type is {x['sugar_type']}. Temperature (C) is {x['temprature_c']}. Temperature (F) is {x['temprature_f']}. Weight (kg) is {x['weight_kg']}. Pulse is {x['pulse']}. Oxygen is {x['oxygen']}. Allergies is {x['allergies']}.Medicine history is {x['medicine_history']}", 
                lambda x: f"Sex: {x['sex']}. Blood pressure systolic is {x['bps']}. Blood pressure diastolic is {x['bpd']}. Sugar is {x['sugar']}. Sugar Type is {x['sugar_type']}. Temperature (C) is {x['temprature_c']}. Temperature (F) is {x['temprature_f']}. Weight (kg) is {x['weight_kg']}. Pulse is {x['pulse']}. Oxygen is {x['oxygen']}.Patient is allergic from {x['allergies']}.Patient has history of {x['medical_history']}.Patient has history of {x['medical_history']}.Prescribed medicine is {x['medicine_history']}", 
                # " ".join([f"{Allergies} is {Medicine_history}." for Allergies, Medicine_history in zip(x['allergies'], x['medicine_history']) if Allergies is not None and Medicine_history is not None]),
                axis=1
            )
                patient_data = patient_data[['content', 'diagnosis']]
                patient_data = pd.concat([patient_data, new_df], axis=1)
                # print("patient_data ",patient_data)
        else:
            patient_data="No record found"
    else:
         patient_data=" How can we help you"     
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
            template = """ 
            Doctor: Act like you are a doctor and ask questions using patient history.
            Doctor will ask questions about the patient's vitals and conditions at first response.
            Doctor will ask about symptoms.
            Provide general information and suggestions about patient condition without requiring a specific diagnosis or recommending a visit to a healthcare professional. Focus on general tips, common practices, or lifestyle changes that may help manage or alleviate symptoms. Must avoid responses that insist for medical professional or consultant.
            You must analyze user's chat history to avoid asking same questions over and over again and to come up with a solution.
            If patient asked you to prescribe/suggest medicine then Doctor will prescribe the medicine using {medicine_list}.
            If a response includes a need for serious medical attention with a doctor, recommend them to book an appointment with our professional healthworkers at EZSHIFA.
            You will ALWAYS prescribe medicines from {medicine_list} in a FINAL ANSWER. 
            
            QUESTION: {question}
            =========
            {summaries}
            =========
            FINAL ANSWER:"""
            
            
            PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question","medicine_list"])
            p_data = load_qa_with_sources_chain(OpenAIChat(openai_api_key='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', temperature=0, model_name="gpt-3.5-turbo"), prompt=PROMPT)
            ai_response = p_data({"input_documents": index_object.similarity_search(question, k=4), "question": question,"summaries":conversation,"medicine_list":medicine_list}, return_only_outputs=True)["output_text"]
            # speech_bot=text_to_speech(ai_response,language='en',output_file='output.mp3')
            print(ai_response)
            question_array=question_check(ai_response)
        except AttributeError:
                            ai_response="How can we help you:"
        return ai_response,question_array





def question_check(ai_response):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s*', ai_response)
    questioning_words = [
    "Who", "What", "When", "Where", "Why", "How", "Which", "Whose", "Whom",
    "Is", "Are", "Do", "Does", "Did", "Will", "Would", "Can", "Could","If you have any",
    "Should", "May", "Might", "Have", "Has", "Had", "Whether", 
    "What if", "Which one", "Whose", "In what way", "To what extent", 
    "Wherefore", "In what manner", "For what reason", "In what respect", 
    "By what means", "On what grounds", "Under what circumstances", 
    "By what method", "With what intention", "Through what process", 
    "For what purpose", "What on earth", "What the heck", "What in the world", 
    "How come", "What else", "What about", "What for", "What next", "Additionally, have you"
]

    multiple_questions = [sentence.strip() for sentence in sentences if re.search(r'(?:(?<=^)|(?<=\n))(?:\d+\.\s.*?)(?=\n\d+\.|\Z)|\b(?:'+'|'.join(questioning_words)+')\s.*?[?.!]\s*', sentence)]
    question_array=[]
    for i, ques in enumerate(multiple_questions, start=1):
        formated_question=f"Question {i}: {ques}"
        question_array.append(formated_question)
        print(f"Question {i}: {ques}")

    return question_array


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app.run(host='0.0.0.0')
