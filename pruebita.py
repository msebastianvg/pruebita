import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

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

def formatear_miles(numero):
    return '{:,.0f}'.format(numero).replace(',', '.')


st.title('Reporte BETS - 2023')

st.subheader('Comienza el último periodo del año: 09 de Octubre hasta 31 de Diciembre.')


# Cargar los datos desde el archivo Excel
file_path = 'bets-2023-2.xlsx'
df = pd.read_excel(file_path, sheet_name='bets')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by='DATE')

# Encontrar el valor máximo de 'ID' para cada fecha
df['Max_ID'] = df.groupby('DATE')['ID'].transform('max')

# Crear un DataFrame con el último valor de "PERCENTAGE" de cada día
last_pozo_actual = df.groupby('DATE')['PERCENTAGE'].last().reset_index()

# Aplicar la lógica de colores en función de 'WL'
last_wl = df[df['ID'] == df['Max_ID']]
last_wl['Color'] = 'lightgreen'
last_wl.loc[last_wl['WL'] == 0, 'Color'] = 'mistyrose'

# Combinar los DataFrames 'last_pozo_actual' y 'last_wl' para tener los colores
last_pozo_actual = last_pozo_actual.merge(last_wl[['DATE', 'Color']], on='DATE', how='left')
last_pozo_actual['DATE'] = last_pozo_actual['DATE'].dt.strftime('%d-%m-%Y')

fig = px.bar(
    last_pozo_actual,
    x='DATE',
    y='PERCENTAGE',
    color='Color',
    color_discrete_map={'lightgreen': 'lightgreen', 'mistyrose': 'mistyrose'},
)

fig.update_yaxes(
    ticksuffix="%",
    range=[0, 5]
)

fig.update_layout(
    xaxis_title='Fecha',
    yaxis_title='Porcentaje de ganancias (%)',
    xaxis=dict(
        type='category',
        categoryorder='category ascending'  # Ordenar las fechas de manera ascendente
    ),
    showlegend=False
)

st.plotly_chart(fig)



# Crear una columna
col1, col2 = st.columns(2)

# Definir los valores para el panel métrico
label = "Gain"
value = 5000
delta = 1000

# Crear el panel métrico
col1.metric(label=label, value=value, delta=delta)






total_wins = (df['WL'] == 1).sum()
total_losses = (df[df['WL'] == 0]['WL'] == 0).sum()
media = total_wins/(total_losses+total_wins)
apuestas_ganadas = total_wins
apuestas_perdidas = total_losses

fig = px.bar(
    x=["Apuestas ganadas", "Apuestas perdidas"],
    y=[total_wins, total_losses],
    color=["Ganadas", "Perdidas"],
    labels={"x": "", "y": ""},
#    title="Apuestas Ganadas vs. Apuestas Perdidas",
    color_discrete_map={"Ganadas": "lightgreen", "Perdidas": "mistyrose"}
)
st.write(fig)


# Cargar los datos desde el archivo Excel
file_path = 'bets-2023-2.xlsx'
sheet_name = 'bets'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Agrupar por 'CATEGORY' y 'WL' y contar la cantidad de registros
grouped = df.groupby(['CATEGORY', 'WL']).size().unstack(fill_value=0)

# Reiniciar el índice para tener 'CATEGORY' como una columna
grouped = grouped.reset_index()

# Cambiar los nombres de las columnas
grouped = grouped.rename(columns={0: 'Apuestas perdidas', 1: 'Apuestas ganadas'})

# Crear un gráfico interactivo en Streamlit
fig = px.bar(
    grouped,
    x='CATEGORY',
    y=['Apuestas perdidas', 'Apuestas ganadas'],  # Corregir los nombres de las columnas
    barmode='group',
    # title='Cantidad de Registros por Categoría y WL',
    color_discrete_map={"Apuestas ganadas": "lightgreen", "Apuestas perdidas": "mistyrose"},  # Corregir la paleta de colores
)
fig.update_layout(
    xaxis_title='Categoría',
    yaxis_title='Cantidad de apuestas',
    showlegend=True,
)

st.plotly_chart(fig)





# Cargar los datos desde el archivo Excel
file_path = 'bets-2023-2.xlsx'
sheet_name = 'bets'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Agrupar por 'CATEGORY' y 'WL' y contar la cantidad de registros
grouped = df.groupby(['TEAM', 'WL']).size().unstack(fill_value=0)

# Reiniciar el índice para tener 'CATEGORY' como una columna
grouped = grouped.reset_index()

# Cambiar los nombres de las columnas
grouped = grouped.rename(columns={0: 'Apuestas perdidas', 1: 'Apuestas ganadas'})

# Crear un gráfico interactivo en Streamlit
fig = px.bar(
    grouped,
    x='TEAM',
    y=['Apuestas perdidas', 'Apuestas ganadas'],  # Corregir los nombres de las columnas
    barmode='group',
    # title='Cantidad de Registros por Categoría y WL',
    color_discrete_map={"Apuestas ganadas": "lightgreen", "Apuestas perdidas": "mistyrose"},  # Corregir la paleta de colores
)
fig.update_layout(
    xaxis_title='Equipo',
    yaxis_title='Cantidad de apuestas',
    showlegend=True,
)

st.plotly_chart(fig)






st.subheader('Monto personal')
input_text = st.text_input("Ingresa tu palabra ultra secreta y presiona Enter:")

df = pd.read_excel(file_path, sheet_name='resumen')
filtro_tipo_2 = df[df['TIPO'] == 2]
if not filtro_tipo_2.empty:
    valor_v = filtro_tipo_2.iloc[0]['V']
    valor_c = filtro_tipo_2.iloc[0]['C']
    valor_e = filtro_tipo_2.iloc[0]['E']
    valor_m = filtro_tipo_2.iloc[0]['M']
    valor_cc = filtro_tipo_2.iloc[0]['CC']

# Verificar si la palabra ingresada es 'lokura'
if input_text.lower() == 'ornn':
    valor_v_v = f"${valor_v:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_v_v.replace(',', '.')}")  
elif input_text.lower() == 'rufi':
    valor_e_e = f"${valor_e:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_e_e.replace(',', '.')}")  
elif input_text.lower() == 'morty':
    valor_c_c = f"${valor_c:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_c_c.replace(',', '.')}")  
elif input_text.lower() == 'duskelokura':
    valor_m_m = f"${valor_m:,.0f} CLP"
    st.write(f"Tu monto a la fecha es de: {valor_m_m.replace(',', '.')}")  
else:
    resultado = 0



st.subheader(' ')
st.subheader(' ')

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





df['DATE'] = pd.to_datetime(df['DATE'])

# Seleccionar el último registro de cada fecha
df = df.groupby('DATE').tail(1)

# Calcular el promedio global de 'PERCENTAGE'
average_single_global = df['PERCENTAGE'].mean()

# Asignar colores según 'WL'
color_map = {'0': 'red', 1: 'green'}

# Crear un gráfico interactivo utilizando Plotly Express
fig = px.bar(df, x='DATE', y='PERCENTAGE', color='WL', color_discrete_map=color_map, labels={'PERCENTAGE': 'GANANCIAS (%)'})
fig.update_traces(marker_line_width=0)  # Eliminar las líneas de borde
fig.update_layout(
    xaxis_title='FECHA',
    yaxis_title='GANANCIAS (%)',
    yaxis_tickformat='%',
    title='GANANCIAS (%) POR FECHA',
    showlegend=True,
)
fig.add_hline(y=average_single_global, line_dash='dash', line_color='blue', name='PROMEDIO DE GANANCIAS (%)')

# Mostrar el gráfico interactivo en Streamlit
st.plotly_chart(fig)






min_single = df['PERCENTAGE'].min()
max_single = df['PERCENTAGE'].max() + 0.01
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.groupby('DATE').tail(1)
average_single_global = df['PERCENTAGE'].mean()
st.write("Porcentaje de ganancias")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['lightgreen' if wl == 1 else 'orange' for wl in df['WL']]
ax.bar(df['DATE'], df['PERCENTAGE'], color=colors)
ax.set_xlabel('FECHA')
ax.set_ylabel('GANANCIAS (%)')
ax.yaxis.set_major_formatter('{:.0%}'.format)
ax.set_title('GANANCIAS (%) POR FECHA')
ax.set_ylim([min_single, max_single])
plt.xticks(rotation=45)
ax.axhline(average_single_global, color='blue', linestyle='--', label='PROMEDIO DE GANANCIAS (%)')
ax.legend()
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









