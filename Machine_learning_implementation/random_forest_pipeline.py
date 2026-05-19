import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

warnings.filterwarnings('ignore')

class PipelineSinaisVitaisRF:
    def __init__(self, pasta_dados='data', arq_com_label='02_treino_sinais_vitais_com_label.txt', arq_sem_label='01_treino_sinais_vitais_sem_label.txt'):
        self.pasta_dados = pasta_dados
        self.arq_com_label = arq_com_label
        self.arq_sem_label = arq_sem_label
        
        # Estruturas de dados internas
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.melhor_modelo = None

    def carregar_e_tratar_dados(self):
        print("\n" + "="*60)
        print(" [FASE 1] CARREGAMENTO E ENGENHARIA DE ATRIBUTOS")
        print("="*60)
        
        caminho_com = os.path.join(self.pasta_dados, self.arq_com_label) if os.path.exists(self.pasta_dados) else self.arq_com_label
        caminho_sem = os.path.join(self.pasta_dados, self.arq_sem_label) if os.path.exists(self.pasta_dados) else self.arq_sem_label
        
        if not os.path.exists(caminho_com):
            raise FileNotFoundError(f"Arquivo contendo as labels não encontrado: {caminho_com}")

        print(f"[-] Carregando base de treino rotulada: {caminho_com}")
        # header=None pois os txts começam direto nos dados numéricos
        df_rotulado = pd.read_csv(caminho_com, header=None, sep=',')
        
        print(f"    -> Dimensões brutas encontradas: {df_rotulado.shape}")
        
        # Coluna 0 = ID sequencial (deve ser descartada para evitar overfitting)
        # Colunas 1 até 6 = Características reais dos Sinais Vitais (Features)
        # Coluna 7 = Alvo/Classe (Target)
        
        self.X = df_rotulado.iloc[:, 1:7] # Seleciona apenas as colunas de sinais vitais
        self.y = df_rotulado.iloc[:, 7].astype(int)   # Seleciona a última coluna como classe alvo
        
        print(f"[+] Ruído de ID removido. Quantidade de Predritores (Features): {self.X.shape[1]}")
        print(f"[-] Distribuição volumétrica das classes na base:")
        for classe, qtd in self.y.value_counts().items():
            print(f"    -> Classe {classe}: {qtd} registros de pacientes.")

        # Divisão Holdout em Treino (70%) e Teste (30%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        print(f"[+] Dados divididos com sucesso: Treino={self.X_train.shape[0]} | Teste={self.X_test.shape[0]}")

    def otimizar_e_treinar(self):
        print("\n" + "="*60)
        print(" [FASE 2] OTIMIZAÇÃO DE HIPERPARÂMETROS (GRID SEARCH)")
        print("="*60)
        
        # Mapeamento de busca focado na teoria de árvores de decisão
        param_grid = {
            'n_estimators': [50, 100, 200],         # Quantidade de árvores no comitê
            'criterion': ['gini', 'entropy'],       # Métrica matemática de ganho de informação
            'max_depth': [None, 10, 15],            # Poda vertical para evitar sobreajuste
            'min_samples_split': [2, 5]             # Controle de divisão de nós internos
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        cv_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("[-] Iniciando buscas exaustivas pelas melhores árvores...")
        inicio = time.time()
        grid_search = GridSearchCV(
            estimator=rf_base, param_grid=param_grid, 
            cv=cv_estratificado, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        self.melhor_modelo = grid_search.best_estimator_
        print(f"[+] Processo concluído em {time.time() - inicio:.2f} segundos.")
        print("\n>>> CONFIGURAÇÃO ÓTIMA DA FLORESTA ENCONTRADA:")
        for param, valor in grid_search.best_params_.items():
            print(f"    * {param}: {valor}")
        print(f"    * Acurácia Média na Validação Cruzada: {grid_search.best_score_:.4f}")

    def avaliar_performance(self):
        print("\n" + "="*60)
        print(" [FASE 3] EXTRACÃO DE MÉTRICAS MULTICRITERIO (TESTE)")
        print("="*60)
        
        # Predições no conjunto de teste isolado
        y_pred = self.melhor_modelo.predict(self.X_test)
        y_prob = self.melhor_modelo.predict_proba(self.X_test)
        
        acuracia = accuracy_score(self.y_test, y_pred)
        precisao = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        auc_roc = roc_auc_score(self.y_test, y_prob, multi_class='ovr', average='weighted')
        
        print(f"[+] Resultados Consolidados no Conjunto Inédito:")
        print(f"    - Acurácia:   {acuracia:.4f} ({acuracia*100:.2f}%)")
        print(f"    - Precisão:   {precisao:.4f}")
        print(f"    - Recall:     {recall:.4f}")
        print(f"    - F1-Score:   {f1:.4f}")
        print(f"    - AUC-ROC:    {auc_roc:.4f}")
        
        print("\n[-] Matriz de Confusão:")
        print(confusion_matrix(self.y_test, y_pred))
        
        print("\n[-] Relatório Detalhado por Classe de Sinais Vitais:")
        print(classification_report(self.y_test, y_pred))
        
        print("-" * 50)
        print(" Relevância dos Sinais Vitais para o Modelo:")
        importancias = self.melhor_modelo.feature_importances_
        for i, imp in enumerate(importancias):
            print(f"    Sinal Vital (Coluna {i+1}) -> Importância Relativa: {imp:.4f}")
        print("="*60)

    def inferir_dados_sem_label(self):
        print("\n" + "="*60)
        print(" [FASE 4] INFERÊNCIA EM DADOS NOVOS (SEM LABEL)")
        print("="*60)
        
        caminho_sem = os.path.join(self.pasta_dados, self.arq_sem_label) if os.path.exists(self.pasta_dados) else self.arq_sem_label
        
        if not os.path.exists(caminho_sem):
            print(f"[AVISO] Arquivo {self.arq_sem_label} não encontrado para inferência.")
            return
            
        print(f"[-] Carregando dados sem rótulo para predição: {caminho_sem}")
        df_sem = pd.read_csv(caminho_sem, header=None, sep=',')
        
        # Isolar os IDs originais e capturar apenas os mesmos preditores do treino (Colunas 1 a 6)
        ids_originais = df_sem.iloc[:, 0]
        X_novos = df_sem.iloc[:, 1:7]
        
        print("[-] Executando predições com o modelo otimizado...")
        predicoes_finais = self.melhor_modelo.predict(X_novos)
        
        df_resultados = pd.DataFrame({
            'ID_Paciente': ids_originais,
            'Classe_Predita': predicoes_finais
        })
        
        nome_saida = 'predicoes_sinais_vitais.txt'
        df_resultados.to_csv(nome_saida, index=False, sep=',')
        print(f"[SUCESSO] Arquivo '{nome_saida}' gerado com as predições salvos para entrega!")
        print("="*60)

    def executar(self):
        self.carregar_e_tratar_dados()
        self.otimizar_e_treinar()
        self.avaliar_performance()
        self.inferir_dados_sem_label()

if __name__ == "__main__":
    pipeline = PipelineSinaisVitaisRF(
        pasta_dados='data', 
        arq_com_label='02_treino_sinais_vitais_com_label.txt',
        arq_sem_label='01_treino_sinais_vitais_sem_label.txt'
    )
    pipeline.executar()