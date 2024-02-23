import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

def load_data():
    # Carregar dados da base
    url = "https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"
    data = pd.read_csv(url)

    # Ajustar os dados (remover colunas irrelevantes e tratar valores faltantes se houver)
    data.dropna(inplace=True)
    features = data.drop(['#', 'Name', 'Type 1', 'Type 2'], axis=1)
    labels = data['Type 1']
    return features, labels

def classify_pokemon(model, features):
    expected_features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']
    features_selected = [features[feature] for feature in expected_features]
    prediction = model.predict([features_selected])[0]
    return prediction

def main():
    # Carregar dados
    features, labels = load_data()

    # Criar modelo de Árvore de Decisão
    model = DecisionTreeClassifier(min_samples_split=25, max_depth=9)  # Ajuste dos parâmetros

    # Treinar o modelo final
    model.fit(features, labels)

    # Avaliar o modelo usando validação cruzada de 10-folds
    scores = cross_val_score(model, features, labels, cv=10)
    print("Taxa de acerto média: {:.2f}%".format(scores.mean() * 100))

    # Exibir a árvore de decisão gerada
    tree_rules = export_text(model, feature_names=features.columns.tolist())
    print("\nÁrvore de decisão gerada:")
    print(tree_rules)

    # Coletar atributos do usuário
    print("\nBem-vindo à aplicação de classificação de tipo de Pokémon!")
    print("Insira os atributos do Pokémon para classificação:")

    features_input = {}
    features_input['Total'] = float(input("Total: "))
    features_input['HP'] = float(input("HP: "))
    features_input['Attack'] = float(input("Ataque: "))
    features_input['Defense'] = float(input("Defesa: "))
    features_input['Sp. Atk'] = float(input("Ataque Especial: "))
    features_input['Sp. Def'] = float(input("Defesa Especial: "))
    features_input['Speed'] = float(input("Velocidade: "))
    features_input['Generation'] = float(input("Geração: "))
    features_input['Legendary'] = input("Lendário (True/False): ").lower() == 'true'

    # Classificar qual o tipo do Pokémon
    predicted_type = classify_pokemon(model, features_input)
    print("O tipo do Pokémon é:", predicted_type)

if __name__ == "__main__":
    main()
