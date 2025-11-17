# predicao-reincidencia-chamadas-callcenter
PREDIÇÃO DE REINCIDÊNCIA DE CHAMADAS NO TELEATENDIMENTO COM APRENDIZADO DE MÁQUINA BASEADO NO COMPORTAMENTO DO CLIENTE

O atendimento telefônico permanece como um dos principais canais de contato entre clientes e empresas, especialmente em setores essenciais, como saneamento. Nesse contexto, a reincidência de chamadas representa um desafio, pois eleva custos operacionais e compromete a satisfação dos usuários. Este estudo tem como objetivo desenvolver um modelo preditivo de aprendizado de máquina capaz de estimar se a probabilidade de um cliente realizar uma nova ligação em até 30 minutos, a partir de dados históricos de uma central de atendimento.  O processo envolveu pré-processamento dos registros, criação de variáveis derivadas do comportamento dos clientes e aplicação de algoritmos supervisionados, incluindo Random Forest, XGBoost, CatBoost e LightGBM. A base foi dividida em 80% para treinamento e 20% para validação, com os três meses finais reservados para teste. Os modelos foram avaliados por métricas clássicas de classificação (acurácia, precisão, recall, F1-score e AUC). XGBoost, LightGBM e CatBoost apresentaram melhor desempenho (AUC > 0,87). A engenharia de features gerou ganhos relevantes, sobretudo em Recall e AUC. Conclui-se que a abordagem é estratégica para otimizar call centers, reduzir custos e aprimorar a experiência do cliente.


## Base de Dados

A base de dados utilizada neste projeto é muito grande para ser armazenada no GitHub.  
Você pode baixá-la através do link abaixo:

🔗 **[Download da Base de Dados no Google Drive](https://drive.google.com/drive/folders/1fMhySwB9aHcBD1Xr52lwRfToAVsTjfuz?usp=sharing)**  