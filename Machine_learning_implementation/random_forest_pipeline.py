import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

warnings.filterwarnings('ignore')

class PipelineSinaisVitaisRF:
    def __init__(self, pasta_dados='data', arq_com_label='02_treino_sinais_vitais_com_label.txt', arq_sem_label='01_treino_sinais_vitais_sem_label.txt'):
        self.pasta_dados = pasta_dados
        self.arq_com_label = arq_com_label
        self.arq_sem_label = arq_sem_label
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.melhor_modelo = None

    def carregar_e_tratar_dados(self):
        print("\n=== [FASE 1] CARREGAMENTO E ENGENHARIA DE ATRIBUTOS ===")
        caminho_com = os.path.join(self.pasta_dados, self.arq_com_label) if os.path.exists(self.pasta_dados) else self.arq_com_label
        
        if not os.path.exists(caminho_com):
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {caminho_com}")

        df_rotulado = pd.read_csv(caminho_com, header=None, sep=',')
        self.X = df_rotulado.iloc[:, 3:6] 
        self.y = df_rotulado.iloc[:, 7].astype(int)
        
        print(f"[+] Dados carregados. Preditores: {self.X.shape[1]} | Registros: {self.X.shape[0]}")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        print(f"[+] Divisão Holdout: Treino={self.X_train.shape[0]} | Teste={self.X_test.shape[0]}")

    def otimizar_e_treinar(self):
        print("\n=== [FASE 2] OTIMIZAÇÃO DE HIPERPARÂMETROS (GRID SEARCH) ===")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 15],
            'min_samples_split': [2, 5]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        cv_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("[-] Buscando melhor configuração de parâmetros...")
        inicio = time.time()
        grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=cv_estratificado, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.melhor_modelo = grid_search.best_estimator_
        print(f"[+] Treinamento concluído em {time.time() - inicio:.2f} segundos.")
        print(f"[+] Melhor Acurácia na Validação Cruzada: {grid_search.best_score_:.4f}")

    def avaliar_performance(self):
        print("\n=== [FASE 3] EXTRAÇÃO DE MÉTRICAS NO TESTE ===")
        y_pred = self.melhor_modelo.predict(self.X_test)
        y_prob = self.melhor_modelo.predict_proba(self.X_test)
        
        print(f"-> Acurácia:  {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"-> Precisão:  {precision_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"-> Recall:    {recall_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"-> F1-Score:  {f1_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"-> AUC-ROC:   {roc_auc_score(self.y_test, y_prob, multi_class='ovr', average='weighted'):.4f}")
        
    def gerar_graficos_artigo(self):
        print("\n=== [FASE 4] GERANDO GRÁFICOS PARA O ARTIGO ===")
        y_pred = self.melhor_modelo.predict(self.X_test)
        sns.set_theme(style="whitegrid")
        
        # 1. Matriz de Confusão
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False, annot_kws={"size": 12})
        plt.title('Matriz de Confusão - Random Forest', fontsize=12, pad=12)
        plt.xlabel('Classe Predita', fontsize=10)
        plt.ylabel('Classe Real', fontsize=10)
        plt.tight_layout()
        plt.savefig('figura1_matriz_confusao.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[+] 'figura1_matriz_confusao.png' salva com sucesso.")

        # 2. Importância das Features
        importancias = self.melhor_modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]
        
        nomes_features = [f"Sinal Vital {i+1}" for i in range(len(importancias))]
        nomes_ordenados = [nomes_features[i] for i in indices]
        valores_ordenados = importancias[indices]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=valores_ordenados, y=nomes_ordenados, palette='viridis')
        plt.title('Importância Relativa dos Sinais Vitais', fontsize=12, pad=12)
        plt.xlabel('Importância (Métrica Gini/Entropy)', fontsize=10)
        plt.ylabel('Características', fontsize=10)
        
        for index, value in enumerate(valores_ordenados):
            plt.text(value, index, f' {value:.3f}', va='center', fontsize=10)
            
        plt.tight_layout()
        plt.savefig('figura2_importancia_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[+] 'figura2_importancia_features.png' salva com sucesso.")

    def inferir_dados_sem_label(self):
        print("\n=== [FASE 5] INFERÊNCIA EM DADOS NOVOS ===")
        caminho_sem = os.path.join(self.pasta_dados, self.arq_sem_label) if os.path.exists(self.pasta_dados) else self.arq_sem_label
        
        if not os.path.exists(caminho_sem):
            print(f"[AVISO] Arquivo {self.arq_sem_label} não encontrado para inferência.")
            return
            
        df_sem = pd.read_csv(caminho_sem, header=None, sep=',')
        ids_originais = df_sem.iloc[:, 0]
        X_novos = df_sem.iloc[:, 3:6]        
        predicoes_finais = self.melhor_modelo.predict(X_novos)
        
        df_resultados = pd.DataFrame({'ID_Paciente': ids_originais, 'Classe_Predita': predicoes_finais})
        nome_saida = 'predicoes_sinais_vitais.txt'
        df_resultados.to_csv(nome_saida, index=False, sep=',')
        print(f"[SUCESSO] Arquivo '{nome_saida}' exportado.")

    def executar(self):
        self.carregar_e_tratar_dados()
        self.otimizar_e_treinar()
        self.avaliar_performance()
        self.gerar_graficos_artigo()
        self.inferir_dados_sem_label()

if __name__ == "__main__":
    pipeline = PipelineSinaisVitaisRF(
        pasta_dados='data', 
        arq_com_label='02_treino_sinais_vitais_com_label.txt',
        arq_sem_label='01_treino_sinais_vitais_sem_label.txt'
    )
    pipeline.executar()