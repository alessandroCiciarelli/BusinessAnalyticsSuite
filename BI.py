import streamlit as st
import streamlit.components.v1 as components  
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.utils.testing import ignore_warnings
import pandas as pd
import numpy as np
from pandasql import sqldf
import sweetviz as sv
import base64 
import pyrebase
import tpot
import tabula
import html5lib
import requests
import time
import codecs
import os

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder



########################################LOGIN	DB
firebaseConfig = {
  "apiKey": st.secrets["apiKey"],
  "authDomain": st.secrets["authDomain"],
  "databaseURL": st.secrets["databaseUR"],
  "projectId": st.secrets["projectId"],
  "storageBucket": st.secrets["storageBucket"],
  "messagingSenderId": st.secrets["messagingSenderId"],
  "appId": st.secrets["appId"],
  "measurementId": st.secrets["measurementId"]
};



# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()

################################################à
timestr = time.strftime("%Y%m%d-%H%M%S")
name = ""
crediti_rimasti = 0
st.set_page_config(page_title="AUTO Analisi Esplorativa ( EDA ) by I.A. Italia", page_icon="🔍", layout='wide', initial_sidebar_state='auto')

st.markdown("<center><h1> AUTO BUSINESS INTELLIGENCE <small>by I. A. ITALIA</small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" > <bold>Tutti i tool di Analisi, Pulizia e Visualizzazione Dati in unico Posto <bold>  </bold><p>', unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True, persist=True)
def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(uploaded_file)
    except :
        dataset = pd.read_csv(uploaded_file)
        
    return dataset
		
def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "Note_Analisi_IAITALIA_{}_.txt".format(timestr)
	st.sidebar.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Scarica le tue NOTE !!</a>'
	st.sidebar.markdown(href,unsafe_allow_html=True)
	st.sidebar.subheader("")
	
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
    
def app(dataset):
    # Use the analysis function from sweetviz module to create a 'DataframeReport' object.
    analysis = sv.analyze([dataset,'EDA2'], feat_cfg=sv.FeatureConfig(force_text=[]), target_feat=None)
    analysis.show_html(filepath='EDA2.html', open_browser=False, layout='vertical', scale=1.0)
    HtmlFile = open("EDA2.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code,height=1200, scrolling=True)
    st.markdown(get_binary_file_downloader_html('EDA2.html', 'Report'), unsafe_allow_html=True)
    st.success("Primo Report Generato Con Successo, per scaricarlo clicca il Link quì sopra.")

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display =F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

def rimuoviCredito():
	if(finecredito()):
		st.session_state.count = db.child(st.session_state.id).child("Crediti").get().val()
		st.session_state.count = st.session_state.count - 1
		db.child(st.session_state.id).update({"Crediti":st.session_state.count})
		st.session_state.count = db.child(st.session_state.id).child("Crediti").get().val()

def finecredito():
	cr = db.child(st.session_state.id).child("Crediti").get().val()
	if cr > 0 : return True
	else: return False

def stampacredito():
	cr = db.child(st.session_state.id).child("Crediti").get().val()
	if cr >= 25 : st.success("Hai a disposizione " + str(cr) + " CREDITI ")
	if cr > 10 and cr <25 : st.warning("Hai a disposizione " + str(cr) + " CREDITI ")
	if cr <10 : st.error("ATTENZIONE! Hai a disposizione " + str(cr) + " CREDITI ")

############################################ANALYTIC SUITE
def AnalyticSuite()  :

	uploaded_file = st.file_uploader("Perfavore inserisci quì il file di tipo : xlsx, csv", type=["csv"])

	st.sidebar.subheader("") 
	st.sidebar.subheader("") 
	st.sidebar.subheader("Notepad")
	my_text = st.sidebar.text_area(label="Inserisci quì le tue osservazioni o note!", value="Al momento non hai nesuna nota...", height=30)

	if st.sidebar.button("Salva"):
		text_downloader(my_text)
		
	if uploaded_file is not None:
	    dataset = pd.read_csv(uploaded_file)
	    colonne = list(dataset.columns)
	    options = st.multiselect("Seleziona le colonne che vuoi usare..",colonne,colonne)
	    dataset = dataset[options]
	    gb = GridOptionsBuilder.from_dataframe(dataset)

	    #customize gridOptions
	    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
	    gb.configure_grid_options(domLayout='normal')
	    gridOptions = gb.build()
	    
	    try:
	    	with st.expander("VISUALIZZA DATASET"):
	    		grid_response = AgGrid(
			    dataset, 
			    gridOptions=gridOptions,
			    width='100%',
			    update_mode="MODEL_CHANGED",
			    )
		
	    	with st.expander("STATISICA DI BASE"):
	    		st.write(dataset.describe())
	    except:
		    print("")
		 
	    st.markdown("", unsafe_allow_html=True)
	    task = st.selectbox("Cosa ti serve ?", ["Crea Report Personalizzato", "Scopri il Miglior Algoritmo di ML per i tuoi dati",
		                         "Crea PipeLine ADHOC in Python per i tuoi dati", "Utilizza le Query SQL sui tuoi dati","Pulisci i Miei Dati"])
		                         
		                         
	    if task == "Crea Report Personalizzato":
	    	if(st.button("Genera 2 Report - Costo 1 credito")):
	    		if finecredito() :
			    	pr = ProfileReport(dataset, explorative=True, orange_mode=False)
			    	st_profile_report(pr)
			    	pr.to_file("EDA.html")
			    	st.markdown(get_binary_file_downloader_html('EDA.html', 'Report'), unsafe_allow_html=True)
			    	st.success("Primo Report Generato Con Successo, per scaricarlo clicca il Link quì sopra.")
			    	app(dataset)
			    	rimuoviCredito()
			    	st.balloons()
	    		else:
		    		st.error('Attenzione hai finito i crediti')

	    	
	    elif task == "Scopri il Miglior Algoritmo di ML per i tuoi dati":
	    	datasetMalgo = dataset
	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	    	datasetMalgo = datasetMalgo.select_dtypes(include=numerics)
	    	datasetMalgo = datasetMalgo.dropna()
	    	colonne = datasetMalgo.columns
	    	target = st.selectbox('Scegli la variabile Target', colonne )
	    	st.write("target impostato su " + str(target))
	    	datasetMalgo = datasetMalgo.drop(target,axis=1)
	    	colonne = datasetMalgo.columns
	    	descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
	    	st.write("Variabili Indipendenti impostate su  " + str(descrittori))
	    	
	    	problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
	    	tipo_di_problema = st.selectbox('Che tipo di Algortimo devi utilizzare sui tuoi dati ?', problemi)
	    	percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
		
	    	X = dataset[descrittori]
	    	y = dataset[target]
		
	    	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)
		
	    	if(st.button("Svelami il Miglior Algoritmo per i miei dati- Costo 1 credito")):
	    		if finecredito() :
	    			
		    		if(tipo_di_problema == "CLASSIFICAZIONE"):
		    			from lazypredict.Supervised import LazyClassifier
		    			with st.spinner("Dacci un attimo, stiamo provando tutti gli algoritmi di Classificazione sui tuoi dati"):
			    			clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
			    			models,predictions = clf.fit(X_train, X_test, y_train, y_test)
			    			st.write(models)
			    			models = pd.DataFrame(models)
			    			models.to_csv("model.csv")
			    			st.markdown(get_binary_file_downloader_html('model.csv', 'Rapporto Modelli Predittivi'), unsafe_allow_html=True)
			    			rimuoviCredito()
			    			st.balloons()
			    			
		    		if(tipo_di_problema == "REGRESSIONE"):
		    			from lazypredict.Supervised import LazyRegressor
		    			with st.spinner("Dacci un attimo, stiamo provando tutti gli algoritmi di Regressione sui tuoi dati"):
			    			reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
			    			models, predictions = reg.fit(X_train, X_test, y_train, y_test)
			    			st.write(models)
			    			models = pd.DataFrame(models)
			    			models.to_csv("model.csv")
			    			st.markdown(get_binary_file_downloader_html('model.csv', 'Rapporto Modelli Predittivi'), unsafe_allow_html=True)
			    			rimuoviCredito()
			    			st.balloons()
			    			
	    		else:
	    			st.error('Attenzione hai finito i crediti')
	    			
	    elif task == "Utilizza le Query SQL sui tuoi dati":
	    	q = st.text_input("Scrivi qui dentro la tua Query", value="SELECT * FROM dataset")
	    	if st.button("Applica Query SQL - Costo 1 credito"):
	    		if finecredito() :
		    		df = sqldf(q)
		    		df = pd.DataFrame(df)
		    		st.write(df)
		    		df.to_csv("Dataset_query.csv")
		    		st.markdown(get_binary_file_downloader_html('Dataset_query.csv', 'Riusltato qyery Sql IAITALIA'), unsafe_allow_html=True)
		    		rimuoviCredito()
		    		st.balloons()
		    	else:
		    		st.error('Attenzione hai finito i crediti')
	    
	    elif task == "Crea PipeLine ADHOC in Python per i tuoi dati":
	    	datasetPalgo = dataset
	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	    	datasetPalgo = datasetPalgo.select_dtypes(include=numerics)
	    	datasetPalgo = datasetPalgo.dropna()
	    	colonne = datasetPalgo.columns
	    	target = st.selectbox('Scegli la variabile Target', colonne )
	    	st.write("target impostato su " + str(target))
	    	datasetPalgo = datasetPalgo.drop(target,axis=1)
	    	colonne = datasetPalgo.columns
	    	descrittori =  st.multiselect('Scegli la variabili Indipendenti', colonne )
	    	st.write("Variabili Indipendenti impostate su  " + str(descrittori))
	    	
	    	problemi = ["CLASSIFICAZIONE", "REGRESSIONE" ]
	    	tipo = st.selectbox('Che tipo di Algortimo devi utilizzare sui tuoi dati ?', problemi)
	    	percentuale_dati_test = st.slider('Seleziona la percentuale di dati per il Test', 0.1, 0.9, 0.25)
		
	    	gen = st.slider('GENERAZIONI : Numero di iterazioni del processo di ottimizzazione della pipeline di esecuzione. Deve essere un numero positivo o Nessuno.', 1, 10, 5)
	    	pop = st.slider('POPOLAZIONE : Numero di dati da mantenere nella popolazione di programmazione genetica in ogni generazione.', 1, 150, 20)
	    	
	    	scor = ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'precision']
	    	sel_scor = st.selectbox('Che tipo di metrica vuoi che sia utilizzato ? Se non conosci questi metodi inserisci "accuracy"', scor)
		
	    	X = dataset[descrittori]
	    	y = dataset[target]
	    	
	    	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentuale_dati_test)
		
	    	if(st.button("Creami la miglior pipeline in Python perfavore - Costo 1 credito")):
	    		if finecredito() :
		    		
			    	if tipo=="CLASSIFICAZIONE":
			    		from tpot import TPOTClassifier
			    		with st.spinner("Dacci qualche minuto, stiamo scrivendo il Codice in Python che implementa il miglior algoritmo sui tuoi dati e ottimizzandolo con gli iperparametri. Maggiore è il numero di Generazioni e Popolazione maggiore sarà il tempo di ATTESA..."):
				    		pipeline_optimizer = TPOTClassifier()
				    		pipeline_optimizer = TPOTClassifier(generations=gen, population_size=pop, scoring=sel_scor, cv=5,
								    random_state=42, verbosity=2)
				    		pipeline_optimizer.fit(X_train, y_train)
				    		st.write(f"Accuratezza PIPELINE : {pipeline_optimizer.score(X_test, y_test)*100} %")
				    		pipeline_optimizer.export('IAITALIA_exported_pipeline.py')
				    		filepipeline = open("IAITALIA_exported_pipeline.py", 'r', encoding='utf-8')
				    		source_code = filepipeline.read() 
				    		st.subheader("Miglior PipeLine Rilevata Sui tuoi Dati ")
				    		my_text = st.text_area(label="Hai visto, Scriviamo anche il codice al posto tuo...", value=source_code, height=500)

				    		st.markdown(get_binary_file_downloader_html('IAITALIA_exported_pipeline.py', 'pipeline.py IAITALIA'), unsafe_allow_html=True)
				    		rimuoviCredito()
				    		st.balloons()
			    		
			    	if tipo=="REGRESSIONE":
			    		from tpot import TPOTRegressor
			    		with st.spinner(" Dacci qualche minuto, stiamo scrivendo il Codice in Python che implementa il miglior algoritmo sui tuoi dati e ottimizzandolo con gli iperparametri. Maggiore è il numero di Generazioni e Popolazione maggiore sarà il tempo di ATTESA..."):
				    		pipeline_optimizer = TPOTRegressor()
				    		pipeline_optimizer = TPOTRegressor(generations=gen, population_size=pop, scoring=sel_scor, cv=5,
								    random_state=42, verbosity=2)
				    		pipeline_optimizer.fit(X_train, y_train)
				    		st.write(f"Accuratezza PIPELINE : {pipeline_optimizer.score(X_test, y_test)*100} %")
				    		pipeline_optimizer.export('IAITALIA_exported_pipeline.py')
				    		filepipeline = open("IAITALIA_exported_pipeline.py", 'r', encoding='utf-8')
				    		source_code = filepipeline.read() 
				    		st.subheader("Miglior PipeLine Rilevata Sui tuoi Dati ")
				    		my_text = st.text_area(label="Hai visto, Scriviamo anche il codice al posto tuo...", value=source_code, height=500)

				    		st.markdown(get_binary_file_downloader_html('IAITALIA_exported_pipeline.py', 'pipeline.py IAITALIA'), unsafe_allow_html=True)
				    		rimuoviCredito()
				    		st.balloons()
			    
	    		else:
		    		st.error('Attenzione hai finito i crediti')
		    	
		    	
	    elif task == "Pulisci i Miei Dati":
	    	from datacleaner import autoclean
	    	dataset_pulito=dataset
	    	st.subheader("Ecco qualche INFO sul tuo Dataset Prima che venga pulito")
	    	import io 
	    	buffer = io.StringIO() 
	    	dataset.info(buf=buffer)
	    	s = buffer.getvalue() 
	    	with open("df_info.txt", "w", encoding="utf-8") as f:
	    	     f.write(s) 
	    	fileinfo = open("df_info.txt", 'r', encoding='utf-8')
	    	source_code = fileinfo.read() 
	    	st.text_area(label="info...", value=source_code, height=300)
	    	
	    	col1, col2, col3 = st.columns(3)
	    	
	    	if( col1.button("Pulisci i miei dati da Valori nulli o corrotti - Costo 1 credito")):
	    		if finecredito() :
	    			with st.spinner(" Dacci qualche secondo per ripulire i tuoi dati da Valori NULL o NAN e quelli corrotti "):
			    	    	st.subheader("Ecco qualche INFO sul tuo Dataset Dopo essere stato Pulito")
			    	    	dataset_pulito=autoclean(dataset)
			    	    	buffer = io.StringIO() 
			    	    	dataset.info(buf=buffer)
			    	    	s = buffer.getvalue() 
			    	    	with open("df_info.txt", "w", encoding="utf-8") as f:
			    	    	     f.write(s) 
			    	    	fileinfo = open("df_info.txt", 'r', encoding='utf-8')
			    	    	source_code = fileinfo.read() 
			    	    	st.text_area(label="info dati puliti...", value=source_code, height=300)
			    	    	dataset_pulito.to_csv('I_tuoi_dati_puliti_by_IAITALIA.csv', sep=',', index=False)
			    	    	st.markdown(get_binary_file_downloader_html('I_tuoi_dati_puliti_by_IAITALIA.csv', 'Dati puliti by IAITALIA'), unsafe_allow_html=True)
			    	    	rimuoviCredito()
			    	    	st.balloons()
	    		else:
		    	    	st.error('Attenzione hai finito i crediti')
		    	    	
	    	if( col2.button("Normalizza i valori Numerici [MINMAXSCALER] - Costo 1 credito")):
	    		if finecredito() :
	    			with st.spinner(" Dacci qualche secondo per Normalizzare i tuoi dati "):
			    	    	datasetMM = dataset
			    	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
			    	    	datasetMM = datasetMM.select_dtypes(include=numerics)
			    	    	datasetMM = datasetMM.dropna()
			    	    	from sklearn.preprocessing import MinMaxScaler
			    	    	scaler = MinMaxScaler()
			    	    	scaled = scaler.fit_transform(datasetMM)
			    	    	colonneMM = datasetMM.columns
			    	    	scaled = pd.DataFrame(scaled, columns = colonneMM)
			    	    	st.write(scaled)
			    	    	scaled.to_csv('I_tuoi_dati_MINMAXSCALER_by_IAITALIA.csv', sep=',', index=False)
			    	    	st.markdown(get_binary_file_downloader_html('I_tuoi_dati_MINMAXSCALER_by_IAITALIA.csv', 'Dati normalizzati con metodo MINMAXSCALER by IAITALIA'), unsafe_allow_html=True)
			    	    	rimuoviCredito()
			    	    	st.balloons()
	    		else:
		    	    	st.error('Attenzione hai finito i crediti')
		    	    	
		    	    	
	    	if( col3.button("Standadizza i valori Numerici [STANDARSCALER] - Costo 1 credito")):
	    		if finecredito() :
	    			with st.spinner(" Dacci qualche secondo per Standardizzare i tuoi dati "):
			    	    	datasetSS = dataset
			    	    	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
			    	    	datasetSS = datasetSS.select_dtypes(include=numerics)
			    	    	datasetSS = datasetSS.dropna()
			    	    	from sklearn.preprocessing import MinMaxScaler
			    	    	scaler = MinMaxScaler()
			    	    	scaled = scaler.fit_transform(datasetSS)
			    	    	colonneSS = datasetSS.columns
			    	    	scaled = pd.DataFrame(scaled, columns = colonneSS)
			    	    	st.write(scaled)
			    	    	scaled.to_csv('I_tuoi_dati_STANDARSCALER_by_IAITALIA.csv', sep=',', index=False)
			    	    	st.markdown(get_binary_file_downloader_html('I_tuoi_dati_STANDARSCALER_by_IAITALIA.csv', 'Dati normalizzati con metodo STANDARSCALER by IAITALIA'), unsafe_allow_html=True)
			    	    	rimuoviCredito()
			    	    	st.balloons()
	    		else:
		    	    	st.error('Attenzione hai finito i crediti')




###########################WEBSCRAPESUITE
def ScrapeSuite():

	st.sidebar.subheader("") 
	st.sidebar.subheader("") 
	st.sidebar.subheader("Notepad")
	my_text = st.sidebar.text_area(label="Inserisci quì le tue osservazioni o note!", value="Al momento non hai nesuna nota...", height=30)

	if st.sidebar.button("Salva"):
		text_downloader(my_text)
		
	st.subheader("") 
	st.markdown("### **1️⃣ Inserisci l'url di una pagina web contenente almeno una Tabella **")
	
	try:
	    url =  st.text_input("", value='https://www.tuttosport.com/live/classifica-serie-a', max_chars=None, key=None, type='default')
	    if url and st.button("Cerca le tabelle nella pagina per poi scaricarle - Costo 1 credito"):
	    	if finecredito() :
	    		rimuoviCredito()
		    	arr = ['https://', 'http://']
		    	if any(c in url for c in arr):
		    	    header = {
			    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
			    "X-Requested-With": "XMLHttpRequest"
			    }

		    	    @st.cache(persist=True, show_spinner=False)
		    	    def load_data():
		    	        r = requests.get(url, headers=header)
		    	        return pd.read_html(r.text)

		    	    df = load_data()

		    	    length = len(df)
			    
		    	    if length == 1:
		    	        st.write("Questa Pagina Web Contiene una sola Pagina Web" )
		    	    else: st.write("Questa Pagina Web Contiene " + str((length-1)) + " Tabelle" )


		    	    st.subheader("") 
		    	    def createList(r1, r2): 
		    	        return [item for item in range(r1, r2)] 
			       
		    	    r1, r2 = 0, length
		    	    funct = createList(r1, r2)
		    	    st.markdown('### **2️⃣ Seleziona la tabella che desideri esportare **')
		    	    for i in funct:
			    ###### Selectbox - Selectbox - Selectbox - Selectbox - Selectbox - Selectbox - Selectbox -
		    	    	with st.expander("Tabella numero : " + str(i)): 
		    	    		df1 = df[i]
		    	    		if df1.empty:
		    	    			st.warning ('ℹ️ - Mannaggia! Qualcosa è andato storto..')
		    	    		else:
		    	    			df1 = df1.replace(np.nan, 'empty cell', regex=True)
		    	    			st.dataframe(df1)
		    	    			try:
	    	    					nome_web=csv = "web_table_"+str(i)+".csv"
	    	    					csv = df1.to_csv(index=False)
	    	    					b64 = base64.b64encode(csv.encode()).decode()
	    	    					if st.button("Scarica tabella numero " + str(i) and finecredito()==True):
		    	    					st.markdown('### ** ⬇️ Scarica la tabella in formato csv **')
		    	    					href = f'<a href="data:file/csv;base64,{b64}" download="web_table.csv">** Clicca Qui per Scaricare il Tuo Dataset! 🎉**</a>'
		    	    					st.markdown(href, unsafe_allow_html=True)
		    	    					rimuoviCredito()
		    	    					st.balloons()
		    	    				else:
		    	    					st.error('Attenzione hai finito i crediti')


		    	    			except ValueError:
		    	    				pass
		    	    #ValueSelected = st.selectbox('', funct)
		    	    #st.write('Hai selezionato la Tabella #', ValueSelected)
		    	    
				
			    #df.columns = df.columns.str.replace(r"[()]", "_
			    #df2 = df1.val.replace({'vte':'test'}, regex=True)
	    	
		    	else:
		    		st.error ("⚠️ - L'URL deve avere un formato valido, Devi iniziare con *https://* o *http://*")    
    		else:
	    		st.error('Attenzione hai finito i crediti')
	except :
	    st.info ("ℹ️ - Non abbiamo trovato tabelle da Esportare ! 😊")



###########################PDFTOCSV
def pdftocsv():
	st.subheader("") 
	st.markdown("### **1️⃣ Carica il Tuo PDF **")
	uploaded_file = st.file_uploader('Scegli un file con estensione .pdf contenente almeno una tabella', type="pdf")
	if uploaded_file is not None:
		displayPDF(uploaded_file)
		if st.button("Fammi vedere Che riesci a fare.. - Costo 1 credito"):
			if finecredito() :
				try:
					#df = read_pdf(uploaded_file, pages='all')[0]
					tables = tabula.read_pdf(uploaded_file, pages='all')
					j=0
					for tabelle in tables :
						try:
							if not tabelle.empty :
								j=j+1
								print(tabelle)
								with st.expander("Tabella numero : " + str(j) ):
									df_temp = pd.DataFrame(tabelle)
									df_temp = df_temp.dropna()
									st.write(df_temp)
									csv = df_temp.to_csv(index=False)
									b64 = base64.b64encode(csv.encode()).decode()
									if st.button("Scarica Tabella numero " + str(j) and finecredito()==True):
										st.markdown('### ** ⬇️ Scarica la tabella in formato csv **')
										href = f'<a href="data:file/csv;base64,{b64}" download="PDF_table{str(j)}.csv">** Clicca Qui per Scaricare il Tuo Dataset! 🎉**</a>'
										st.markdown(href, unsafe_allow_html=True)
									else:
										st.error('Attenzione hai finito i crediti')
						except ValueError:
							pass
				except ValueError:
					st.info ("ℹ️ - Non abbiamo trovato tabelle da Esportare ! 😊")
			else:
				st.error('Attenzione hai finito i crediti')

#################MAIN

def main():
	st.subheader("Benvenuto "+ str(st.session_state.key) )
	stampacredito()
	Menu = st.selectbox("Menu", ["Analizza i Tuoi File CSV o Excel - Analytic Suite", "Scarica Tabelle da Pagine web - WebScrape Siute", "Trasforma i tuoi pdf in file csv da analizzare"])


	if Menu == "Analizza i Tuoi File CSV o Excel - Analytic Suite" :
		AnalyticSuite()
	if Menu == "Scarica Tabelle da Pagine web - WebScrape Siute" :
		ScrapeSuite()
	if Menu == "Trasforma i tuoi pdf in file csv da analizzare" :
		pdftocsv()



def login():
	

	

	st.sidebar.title("Business Intelligence Suite")

	# Authentication
	choice = st.sidebar.selectbox('Che devi fare', ['LOGIN', 'Registrazione'])

	# Obtain User Input for email and password
	email = st.sidebar.text_input('Inserisci la tua email')
	password = st.sidebar.text_input('Inserisci la tua password',type = 'password')

	# App 

	# Sign up Block
	if choice == 'Registrazione':
	    handle = st.sidebar.text_input('Perfavore inserisci un NickName', value='')
	    submit = st.sidebar.button('Crea il mio Account')

	    if submit:
	    	try:
		    	user = auth.create_user_with_email_and_password(email, password)
		    	st.success("L'Account è stato Creato Con successo")
		    	st.warning("Per iniziare ad usare la Suite vai nella tua email e conferma l'indirizzo usato per creare l'account")
		    	st.success("Solo dopo aver effettuato la verifica, torna sul sito ed Effettua il LOGIN, riscattando anche i tuoi primi 50 CREDITI GRATIS")
		    	st.balloons()
			# Sign in
		    	user = auth.sign_in_with_email_and_password(email, password)
		    	db.child(user['localId']).child("Handle").set(handle)
		    	db.child(user['localId']).child("Crediti").set(50)
		    	db.child(user['localId']).child("ID").set(user['localId'])
		    	name = handle
		    	link_verifica = auth.generate_email_verification_link(email, action_code_settings=None)
		    	st.write("link_verifica")
		    	"""
		    	crediti_rimasti = 50
		    	st.session_state.key = name
		    	if 'count' not in st.session_state :
		    		st.session_state.count = crediti_rimasti
		    	if 'id' not in st.session_state :
		    		st.session_state.id = user['localId']
		    	"""
	    	except ValueError:
	    		st.error("Attenzione qualcosa è andato storto, Riprova")


	# Login Block
	if choice == 'LOGIN' and st.sidebar.button("LogIn"):
	    try:
		    user = auth.sign_in_with_email_and_password(email,password)
		    name = db.child(user['localId']).child("Handle").get().val()
		    crediti_rimasti = db.child(user['localId']).child("Crediti").get().val()
		    st.session_state.key = name
		    if 'count' not in st.session_state :
		    	st.session_state.count = crediti_rimasti
		    if 'id' not in st.session_state :
		    		st.session_state.id = user['localId']
	    except:
		    st.error("Attenzione qualcosa è andato storto, Riprova")



if 'key' not in st.session_state :
	login()
	

if 'key' in st.session_state :
	main()
	
    			

		
