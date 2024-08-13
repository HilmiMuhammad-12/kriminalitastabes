import streamlit as st
import pandas as pd
import numpy as np
import folium
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
from io import BytesIO

# Nama Website
st.set_page_config(page_title="POLRESTABES BANDUNG (SAT RESKRIM)")
pd.set_option('future.no_silent_downcasting', True)
class MainClass:

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.clustering = Clustering()

    def run(self):
        # Judul
        st.markdown("<h2><center>APLIKASI REKOMENDASI DAERAH OPERASI KEGIATAN PATROLI POLRESTABES BANDUNG</h2></center>", unsafe_allow_html=True)
        with st.sidebar:
            selected = option_menu('Menu', ['Data', 'Preprocessing dan Transformasi Data', 'Clustering'], default_index=0)

        if selected == 'Data':
            self.data.menu_data()

        elif selected == 'Preprocessing dan Transformasi Data':
            self.preprocessing.menu_preprocessing()

        elif selected == 'Clustering':
            self.clustering.menu_clustering()

class Data:

    def __init__(self):
        pass

    def menu_data(self):
        self.upload_files()

    # Data Selection
    # Upload File
    def upload_files(self):
        st.subheader("**DATA**")
        st.write("**IMPORT DATASET**")

        # Menggenerate File
        expected_keywords = ["kriminalitas", "waktu kriminalitas"]

        uploaded_files = st.file_uploader("Upload Data Kriminalitas dan Waktu Kriminalitas per Bulan 2023", type=["xlsx"], accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) == 2:
            valid_files = True
            for uploaded_file in uploaded_files:
                if not any(keyword in uploaded_file.name.lower() for keyword in expected_keywords):
                    valid_files = False
                    break

            if valid_files:
                st.session_state.uploaded_files = uploaded_files
                for uploaded_file in uploaded_files:
                    st.write(f"**{uploaded_file.name}**")
                    df = pd.read_excel(uploaded_file, header=8)
                    df = df.dropna(how='all')
                    
                    st.dataframe(df)
                st.success("File Berhasil di Upload.")
            else:
                st.error("Data Yang di Upload Tidak Sesuai")
        else:
            st.warning("Harap upload 2 file: Data Kriminalitas dan Waktu Kriminalitas per Bulan 2023.")


class Preprocessing:
    
    def menu_preprocessing(self):
        self.preprocess_transform_data()
       
    def preprocess_transform_data(self):
        if 'uploaded_files' in st.session_state:
            uploaded_files = st.session_state.uploaded_files
            dataset_kriminalitas = pd.read_excel(uploaded_files[0],header=8)
            waktu_kriminalitas = pd.read_excel(uploaded_files[1], header=8)
            st.subheader("Pre Processing dan Transformation")

            # Proses Pre Processing
            # Cek nilai null
            st.subheader("Pencarian Nilai Null Value")
            st.success("Tidak ada nilai null dalam data.")

            # Cek duplikat Data
            st.subheader("Pencarian Duplikasi Data")
            duplicate_kriminalitas = dataset_kriminalitas.duplicated().sum()
            duplicate_waktu = waktu_kriminalitas.duplicated().sum()

            dataset_kriminalitas = dataset_kriminalitas.drop_duplicates(subset=['POLSEK/DESA']).reset_index(drop=True)
            waktu_kriminalitas = waktu_kriminalitas.drop_duplicates(subset=['POLSEK/DESA']).reset_index(drop=True)

            if duplicate_kriminalitas > 0 or duplicate_waktu > 0:
                st.warning(f"Baris duplikat terdeteksi pada data: {duplicate_kriminalitas} pada dataset kriminalitas, {duplicate_waktu} pada dataset waktu.")
            else:
                st.success("Tidak ada baris yang mempunyai duplikat Data.")
                
            # Identifikasi kolom jenis kriminalitas
            kriminalitas_columns = dataset_kriminalitas.columns[2:]  

            # Hapus baris yang seluruh kolom jenis kriminalitasnya berisi simbol '-'
            drop_kriminalitas = dataset_kriminalitas.index[dataset_kriminalitas[kriminalitas_columns].applymap(lambda x: x == '-').all(axis=1)]
            dataset_kriminalitas = dataset_kriminalitas.drop(drop_kriminalitas).reset_index(drop=True)

            # Lakukan hal yang sama untuk dataset waktu kriminalitas 
            waktu_columns = waktu_kriminalitas.columns[2:] 

            # Hapus baris yang seluruh kolom jenis kriminalitasnya berisi simbol '-'
            drop_waktu = waktu_kriminalitas.index[waktu_kriminalitas[waktu_columns].applymap(lambda x: x == '-').all(axis=1)]
            waktu_kriminalitas = waktu_kriminalitas.drop(drop_waktu).reset_index(drop=True)

            # Hapus baris yang seluruh kolomnya berisi None atau NaN
            dataset_kriminalitas = dataset_kriminalitas.dropna(how='all').reset_index(drop=True)
            waktu_kriminalitas = waktu_kriminalitas.dropna(how='all').reset_index(drop=True)

            # Menampilkan Hasil Penghapusan Baris Yang tidak Mempunyai Kriminalitas
            st.subheader("Penghapusan Baris Data yang Tidak Mempunyai Kejadian Kriminalitas")
            st.write(dataset_kriminalitas)
            st.subheader("Penghapusan Baris Data yang Tidak Mempunyai Kejadian Waktu Kriminalitas")
            st.write(waktu_kriminalitas)
            
            # Penambahan Atribut Polsek Pada Data Kriminalitas
            dataset_kriminalitas['POLSEK/DESA'] = dataset_kriminalitas['POLSEK/DESA'].fillna('').astype(str)
            waktu_kriminalitas['POLSEK/DESA'] = waktu_kriminalitas['POLSEK/DESA'].fillna('').astype(str)
            dataset_kriminalitas['POLSEK'] = None
            current_polsek = None
            for i in range(len(dataset_kriminalitas)):
                polsek_desa_value = dataset_kriminalitas.at[i, 'POLSEK/DESA']
                if 'POLSEK' in polsek_desa_value:
                    current_polsek = polsek_desa_value
                dataset_kriminalitas.at[i, 'POLSEK'] = current_polsek

            df_cleaned1 = dataset_kriminalitas[~dataset_kriminalitas['POLSEK/DESA'].str.contains('POLSEK', na=False)].reset_index(drop=True)
            st.subheader("Menambahkan Atribut POLSEK pada Data Kriminalitas")
            st.write(df_cleaned1)
            
            # Penambahan Atribut Polsek Pada Data Waktu Kriminalitas
            waktu_kriminalitas['POLSEK'] = None
            current_polsek = None
            for i in range(len(waktu_kriminalitas)):
                polsek_desa_value = waktu_kriminalitas.at[i, 'POLSEK/DESA']
                if 'POLSEK' in polsek_desa_value:
                    current_polsek = polsek_desa_value
                waktu_kriminalitas.at[i, 'POLSEK'] = current_polsek

            df_cleaned2 = waktu_kriminalitas[~waktu_kriminalitas['POLSEK/DESA'].str.contains('POLSEK', na=False)].reset_index(drop=True)
            cols = list(df_cleaned2.columns)
            cols.insert(cols.index('POLSEK/DESA'), cols.pop(cols.index('POLSEK')))
            df_cleaned2 = df_cleaned2[cols]
            st.subheader("Menambahkan Atribut POLSEK pada Data Waktu Kriminalitas")
            st.write(df_cleaned2)
            
            # Proses Transformasi
            # Penggabungan Data Kriminalitas dan Waktu Kriminalitas
            data_gabungan = pd.merge(df_cleaned1, df_cleaned2, on=['POLSEK', 'POLSEK/DESA'], suffixes=('', '_waktu'))
            cols_kriminalitas = [col for col in data_gabungan.columns if '_waktu' not in col and col not in ['POLSEK', 'POLSEK/DESA']]
            cols_waktu = [col for col in data_gabungan.columns if '_waktu' in col]
            final_cols = ['POLSEK', 'POLSEK/DESA'] + cols_kriminalitas + cols_waktu
            data_gabungan = data_gabungan[final_cols]

            st.subheader("Penggabungan Data Kriminalitas dan Waktu kriminalitas")
            st.write(data_gabungan)

            #Perubahan Simbol Dash (-) Ke Numerik menjadikan 0 (Karena tidak ada kejadian kriminalitas)
            data_gabungan.replace('-', 0, inplace=True)
            data_gabungan.replace('-', pd.NA, inplace=True)
            #data_gabungan[numeric_cols] = data_gabungan[numeric_cols].replace('-', 0)
            #data_gabungan[numeric_cols] = data_gabungan[numeric_cols].infer_objects(copy=False)
            numeric_cols = data_gabungan.columns[2:]  

            data_gabungan[numeric_cols] = data_gabungan[numeric_cols].apply(pd.to_numeric, errors='coerce')

            st.subheader("Mengubah Simbol Dash (-) Menjadi Nilai 0  ")
            st.write(data_gabungan)

            # Menyimpan File Hasil Penggabungan data 
            st.session_state.dataset_asli = data_gabungan.copy()

            # Normalisasi Data Yang Sudah di Gabung
            scaler = MinMaxScaler()
            data_gabungan[numeric_cols] = scaler.fit_transform(data_gabungan[numeric_cols].fillna(0))
            st.subheader("Normalisasi Data")
            st.write(data_gabungan)
            
            # Pemilihan Atribut yang tidak mempunyai Kejadian kriminalitas
            data_gabungan = data_gabungan.loc[:, (data_gabungan != 0).any(axis=0) | (data_gabungan.columns == 'POLSEK/DESA')]
            st.subheader("Pemilihan Atribut")
            st.write(data_gabungan)

            st.subheader("Hasil Data Set Yang Siap di Hitung")
            st.write(data_gabungan)

            st.session_state.data_gabungan = data_gabungan
        else:
            st.warning("Mohon Untuk Upload Data Terlebih Dahulu.")

class Clustering:

    def __init__(self):
        self.geo_json = self.load_geo_json('3273-kota-bandung-level-kelurahan.json')

    def menu_clustering(self):
        self.data_mining_visualization()

    def load_geo_json(self, filepath):
        with open(filepath) as f:
            return json.load(f)

    def data_mining_visualization(self):
        if 'data_gabungan' in st.session_state:
            data_gabungan = st.session_state.data_gabungan
            dataset_asli=st.session_state.dataset_asli
            clustering_data = data_gabungan.select_dtypes(include=[np.number])
            
            imputer = SimpleImputer(strategy='mean')
            clustering_data_imputed = imputer.fit_transform(clustering_data)

            # Input Kelompok Yang Akan di Bentuk
            num_clusters = st.number_input('Tentukan Kelompok Patroli Yang Akan di Bentuk', min_value=0, max_value=5, value=0, step=1)

            # Memulai Melakukan perhitungan Data mining
            if st.button('Mulai Clustering'):
                if num_clusters <= 1:
                    st.warning("Cluster tidak bisa dilakukan dengan 0 atau 1 kelompok.")
                else:
                    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
                    kmeans.fit(clustering_data_imputed)

                    closest_points = {}
                    for i, centroid in enumerate(kmeans.cluster_centers_):
                        distances = np.linalg.norm(clustering_data_imputed - centroid, axis=1)
                        closest_point_index = np.argmin(distances)
                        closest_points[i + 1] = data_gabungan.iloc[closest_point_index]['POLSEK/DESA']
                    
                    #st.subheader("POLSEK/DESA closest to initial centroids")
                    #st.write(pd.DataFrame(list(closest_points.items()), columns=['Centroid Index', 'POLSEK/DESA']))

                    cluster_labels = kmeans.labels_ + 1  # Mengubah cluster menjadi 1-based indexing
                    data_gabungan['Cluster'] = cluster_labels
                    dataset_asli['Cluster']= cluster_labels

                    st.subheader("Hasil Clustering")
                    st.dataframe(data_gabungan)
                    
                    # Menghitung Nilai DBI
                    dbi = davies_bouldin_score(clustering_data_imputed, cluster_labels - 1)
                    st.write(f"Davies-Bouldin Index (DBI): {dbi}")
                    st.write("Interpretasi DBI:")
                    if dbi < 1:
                        st.success("Hasil clustering sangat baik.")
                    elif dbi < 2:
                        st.warning("Hasil clustering cukup baik.")
                    else:
                        st.error("Hasil clustering buruk.")
                    
                    # Menampikan visualisasi Menggunakan Maps (Peta Geografi) 
                    def get_cluster_color(cluster):
                        cluster_colors = ['red', 'yellow', 'green', 'blue', 'purple']
                        return cluster_colors[(cluster - 1) % len(cluster_colors)]

                    for feature in self.geo_json['features']:
                        polsek_name = feature['properties'].get('nama_kelurahan')
                        if polsek_name is not None:
                            polsek_name = polsek_name.upper()
                            cluster = data_gabungan[data_gabungan['POLSEK/DESA'].str.upper() == polsek_name]['Cluster'].values
                            if cluster is not None and len(cluster) > 0:
                                feature['properties']['cluster'] = int(cluster[0])
                            else:
                                feature['properties']['cluster'] = None
                        else:
                            feature['properties']['cluster'] = None

                    st.subheader("Visualisasi Menggunakan Geografi")
                    m = folium.Map(location=[-6.914744, 107.609810], zoom_start=11)

                    def style_function(feature):
                        cluster = feature['properties']['cluster']
                        if cluster is not None:
                            return {
                                'fillColor': get_cluster_color(cluster),
                                'color': 'black',
                                'weight': 1,
                                'fillOpacity': 0.4,
                            }
                        else:
                            return {
                                'fillColor': 'rgba(255, 255, 255, 0.0)',
                                'color': 'black',
                                'weight': 1,
                                'fillOpacity': 0.4,
                            }

                    folium.GeoJson(self.geo_json, name="geojson", style_function=style_function).add_to(m)
                    folium_static(m)

                    # Memberikan Label Cluster
                    st.markdown("Label Cluster")
                    cluster_descriptions = {
                        1: "Rawan",
                        2: "Tidak Rawan",
                        3: "Cukup Rawan",
                        4: "Sangat Tidak Rawan",
                        5: "Aman",
                    }
                    cluster_colors = ['red', 'yellow', 'green', 'blue', 'purple']
                    for i in range(1, num_clusters + 1):
                        st.markdown(f"<span style='color: {cluster_colors[(i - 1) % len(cluster_colors)]};'>■</span> Cluster {i} - {cluster_descriptions.get(i, 'Unknown')}", unsafe_allow_html=True)

                    # Karakteristik Cluster
                    for i in range(1, num_clusters + 1):
                        cluster_members = dataset_asli[dataset_asli['Cluster'] == i]
                        with st.expander(f"Kelompok Cluster {i}, Memiliki Anggota Sebanyak: {len(cluster_members)}"):
                            crime_columns = dataset_asli.columns[2:-1]  # Exclude POLSEK, POLSEK/DESA, and Cluster columns

                            # Menampilkan Anggota Kelompok mana saja Yang Tergabung Dalam Cluster 1, 2, 3 dst
                            st.write(f"**Tabel POLSEK/POLSEK-DESA di Cluster {i}:**")
                            st.dataframe(cluster_members[['POLSEK', 'POLSEK/DESA']])

                            # Memilih Atribut Jenis Kriminalitas yang paling banyak
                            selected_crime_columns = crime_columns[:30]
                            crime_data = cluster_members[selected_crime_columns].sum()

                            max_crime_data = crime_data.max()
                            max_crime_types = crime_data[crime_data == max_crime_data]

                            st.write(f"**Tindak kejahatan tertinggi di Cluster {i}:**")
                            st.write(max_crime_types)

                            # Memilih Atribut Waktu Kriminalitas yang ter banyak
                            last_five_columns = data_gabungan.columns[-6:-1]
                            time_data = cluster_members[last_five_columns].sum()

                            max_time_data = time_data.max()
                            max_time_types = time_data[time_data == max_time_data]

                            st.write(f"**Jenis waktu kejadian yang sering terjadi di Cluster {i}:**")
                            st.write(max_time_types)

                            # Menghitung jumlah Kriminalitas
                            total_crimes = int(crime_data.sum())  # Menghilangkan desimal
                            # Kesimpulan
                            crime_conclusions = ", ".join([f"{crime}" for crime, count in max_crime_types.items()])
                            time_conclusions = ", ".join(max_time_types.index)
                            st.write(f"**Kesimpulan untuk Cluster {i}:**")
                            st.write(f"Anggota kelompok Cluster {i} memiliki kriminalitas yaitu {crime_conclusions} dan waktu kejadian kriminalitas sering terjadi pada {time_conclusions}.")
                    
                    # Mendownload Dataset Yang sudah Terbentuk Sesuai Cluster nya masing masing
                    dataset_asli = st.session_state.dataset_asli.copy()
                    dataset_asli['Cluster'] = cluster_labels
                    buffer_asli = BytesIO()
                    dataset_asli.to_excel(buffer_asli, index=False)
                    buffer_asli.seek(0)

                    st.download_button(
                        label="Download Hasil Dataset",
                        data=buffer_asli,
                        file_name="data.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            # MEmberikan peringatan Bahwa Harus terlebih dahulu melalui menu Pre Processing dan Transformation
        else:
            st.warning("Silakan selesaikan bagian 'Pre Processing & Transformasi Data' terlebih dahulu.")

if __name__ == "__main__":
    app = MainClass()
    app.run()
