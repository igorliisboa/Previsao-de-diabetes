#Importando as bibliotecas
from pkg_resources import run_script
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Título
st.title('Previsão de Diabetes\n')
st.write("""
App que utiliza Machine learning para prever possível diabete dos pacientes\n
Fonte: PIMA - INDIA (Kaggle)
""")

#dataset
df = pd.read_csv('diabetes.csv')

#Cabecalho

st.subheader('Informações dos dados')

#Nome do usuário
user_input = st.sidebar.text_input('Digite seu nome: ')

st.write('Paciente',user_input)

#Dados de entrada
x = df.drop(['Outcome'],1)
y = df['Outcome']
#Separa dados entre treinamento e teste
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)

#Dados dos usuários com a função
def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez',0,15,1)
    glucose = st.sidebar.slider('Glicose',2,200,110)
    blood_pressure = st.sidebar.slider('Pressão sanguínea',0,122,72)
    skin_thickness = st.sidebar.slider('Espessura da pele',0,99,20)
    insulin = st.sidebar.slider('Insulina',0,900,30)
    bmi = st.sidebar.slider('Índice de massa corporal (IMC)',0.0,70.0,15.0)
    dbf = st.sidebar.slider('Histórico familiar de diabete',0.0,3.0,0.0)
    age = st.sidebar.slider('Idade',15,100,21)

    user_data = {
        'Gravidez': pregnancies,
        'Glicose': glucose,
        'Pressão sanguínea': blood_pressure,
        'Espessura de pele': skin_thickness,
        'Insulina': insulin,
        'Índice de massa corporal (IMC)': bmi,
        'Histórico familiar de diabete': dbf,
        'Idade': age
    }

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_data()

#Gráfico
graf = st.bar_chart(user_input_variables)

st.subheader('Dados do usuário')
st.write(user_input_variables)

dtc = DecisionTreeClassifier(criterion='entropy',max_depth=3)
dtc.fit(x_train,y_train)

#Precisão do modelo
st.subheader('Precisão do modelo')
st.write(accuracy_score(y_test,dtc.predict(x_test)) * 100)

#Previsão
prediction =  dtc.predict(user_input_variables)

st.subheader('Previsão: ')
st.write(prediction)
