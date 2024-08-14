import joblib

def predict(feature):

    scaler = joblib.load('scaler.save')

    # X_test_tunggal = color moment
    X_test_tunggal = feature
    X_test_tunggal = scaler.transform(X_test_tunggal) # transform dengan scale data latih di atas

    trash_class = {
        1 : "Cardboard",
        2 : "Glass",
        3 : "Metal",
        4 : "Paper",
        5 : "Plastic"
    }

    model = joblib.load("rbf_1b_2a.pkl")
    result = model.predict(X_test_tunggal)[0]
    
    return trash_class[result]