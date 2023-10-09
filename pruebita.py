import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
# import seaborn as sns
# from sklearn.linear_model import LinearRegression

st.title('Reporte BETS - 2023')
st.subheader('Comienza el último periodo del año: 09 de Octubre hasta 31 de Diciembre.')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data



st.subheader('Resultados del periodo anterior (hasta el 08 de Octubre):')

file_path = 'bets - gh.xlsx'
#file_path = 'https://github.com/msebastianvg/pruebita/blob/726e5dc6981883405d77349748318ec156c82583/bets%20-%20gh.xlsx'  # Replace with the actual path to your Excel file
df = pd.read_excel(file_path)
# st.write(df.head())

total_wins = (df['WL'] == 1).sum()
total_losses = (df[df['WL'] == 0]['WL'] == 0).sum()
#st.subheader('W: ' + str(total_wins))
#st.subheader('L: ' + str(total_losses))
media = total_wins/(total_losses+total_wins)
#st.subheader('Win rate: ' + str("{:.1f}%".format(media * 100)))


apuestas_ganadas = total_wins
apuestas_perdidas = total_losses

fig = px.bar(
    x=["Apuestas ganadas", "Apuestas perdidas"],
    y=[total_wins, total_losses],
    color=["Ganadas", "Perdidas"],
    labels={"x": "", "y": ""},
#    title="Apuestas Ganadas vs. Apuestas Perdidas",
    color_discrete_map={"Ganadas": "lightgreen", "Perdidas": "mistyrose"},
)
st.write(fig)




df['DATE'] = pd.to_datetime(df['DATE'])
apuestas_ganadas = df[df['WL'] == 1]
ganancias_por_dia = apuestas_ganadas.groupby('DATE')['WL'].count()
st.write("Cantidad de Apuestas Ganadas por Día")
st.bar_chart(ganancias_por_dia)





# Cargar los datos desde el archivo Excel
# df = pd.read_excel('E:\\B\\bets.xlsx')

# Calcular el valor mínimo y máximo de la columna 'PERCENTAGE'
min_single = df['PERCENTAGE'].min()
max_single = df['PERCENTAGE'].max() + 0.01

# Convertir la columna 'date' a tipo datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Seleccionar el último registro de cada fecha
df = df.groupby('DATE').tail(1)

# Calcular el promedio global de 'PERCENTAGE'
average_single_global = df['PERCENTAGE'].mean()

# Crear un gráfico de barras que muestra la variación de 'PERCENTAGE' con respecto a las fechas
st.write("Porcentaje de ganancias")

fig, ax = plt.subplots(figsize=(10, 6))

# Asignar colores según 'WL' en el último registro de cada fecha
colors = ['lightgreen' if wl == 1 else 'orange' for wl in df['WL']]

ax.bar(df['DATE'], df['PERCENTAGE'], color=colors)

# Personalizar el gráfico
ax.set_xlabel('FECHA')
ax.set_ylabel('GANANCIAS (%)')

# Establecer el formato de los valores en el eje Y como porcentajes con un decimal
ax.yaxis.set_major_formatter('{:.0%}'.format)

ax.set_title('GANANCIAS (%) POR FECHA')
ax.set_ylim([min_single, max_single])  # Establecer el límite superior del eje Y

# Rotar las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)

# Agregar la línea que muestra el promedio global de 'PERCENTAGE'
ax.axhline(average_single_global, color='blue', linestyle='--', label='PROMEDIO DE GANANCIAS (%)')

# Mostrar la leyenda
ax.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)




# df = pd.read_excel('E:\\B\\bets.xlsx')

reference_date = pd.to_datetime("2023-01-01")  # Establece tu fecha de referencia
df['DAYS_FROM_REFERENCE'] = (df['DATE'] - reference_date).dt.days

# Extrae los datos de 'DAYS_FROM_REFERENCE' y 'SINGLE' del DataFrame
X = df['DAYS_FROM_REFERENCE'].values
Y = df['PERCENTAGE'].values

# Calcula la media de X e Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Calcula la desviación estándar de X e Y
std_X = np.std(X)
std_Y = np.std(Y)

# Calcula el coeficiente de la pendiente (a) y el intercepto (b) de la regresión lineal
a = np.sum((X - mean_X) * (Y - mean_Y)) / np.sum((X - mean_X) ** 2)
b = mean_Y - a * mean_X

# Construye la ecuación lineal Y = aX + b
equation = f"Y = {a:.2f}X + {b:.2f}"

# Imprime la ecuación lineal
print("Ecuación Lineal:")
print(equation)








st.subheader('Monto personal')
input_text = st.text_input("Ingresa tu palabra ultra secreta y presiona Enter:")

# Verificar si la palabra ingresada es 'lokura'
if input_text.lower() == 'lokura':
    resultado = 1
    st.write(f"Tu monto a la fecha es de: {resultado}")
else:
    resultado = 0

