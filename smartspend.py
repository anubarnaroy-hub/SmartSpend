import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

data = pd.read_csv("smartspend_data.csv")

purchase = data.dropna( subset=["price", "left", "need", "result"]).copy()
purchase_inputs = purchase[["price", "left", "need"]]
purchase_labels = purchase["result"]
class_names = ["Buy Now", "Choose Cheaper", "Postpone"]

X_train, X_test, y_train, y_test = train_test_split(purchase_inputs, purchase_labels,
    test_size=0.25,random_state=42, stratify=purchase_labels)

decision_tree = DecisionTreeClassifier( max_depth=5,random_state=42)
decision_tree.fit(X_train, y_train)

test_predictions = decision_tree.predict(X_test)
model_accuracy = accuracy_score(y_test, test_predictions)
model_confusion_matrix = confusion_matrix(y_test, test_predictions, labels=class_names)

decision_tree.fit(purchase_inputs, purchase_labels)

def budget_report(total_money, spending_limit, food, shopping,stationery, other):
    total_spent = food + shopping + stationery + other
    money_left = total_money - total_spent
    limit_left = spending_limit - total_spent
    protected_savings = total_money - spending_limit
    if spending_limit > 0:
        budget_used_percent = total_spent / spending_limit * 100
    else:
        budget_used_percent = 100 if total_spent > 0 else 0
    return {
        "total": total_money,
        "limit": spending_limit,
        "spent": total_spent,
        "money_left": money_left,
        "left": limit_left,
        "savings": protected_savings,
        "budget_used_percent": budget_used_percent
    }

expense_columns = ["food","shopping","stationery","other"]
student_data = data.dropna(subset=expense_columns).copy()
expense_data = student_data[expense_columns]

# Divide amounts by 1000 to keep the input values smaller.
scaled_data = expense_data / 1000

kmeans_model = KMeans(n_clusters=3,random_state=42,n_init=10)
student_data["cluster"] = kmeans_model.fit_predict(scaled_data)
student_data["total"] = student_data[expense_columns].sum(axis=1)

average_spending = student_data.groupby("cluster")["total"].mean().sort_values()

group_labels = ["Low Spending","Medium Spending","High Spending"]
cluster_names = {}
for position, cluster_number in enumerate(average_spending.index):
    cluster_names[cluster_number] = group_labels[position]

def spending_pattern(food, shopping, stationery, other):
    values = [[food / 1000, shopping / 1000, stationery / 1000, other / 1000]]
    new_expenses = pd.DataFrame(values, columns=expense_columns)
    predicted_cluster = kmeans_model.predict(new_expenses)[0]
    return cluster_names[predicted_cluster]

def purchase_recommendation(price, money_left, is_needed, parents_support=False):
    if price > money_left and parents_support:
        return "Buy Now"
    if price > money_left and is_needed:
        return "Choose Cheaper"
    if price > money_left:
        return "Postpone"
    values = [[price, money_left, is_needed]]
    new_purchase = pd.DataFrame(values, columns=["price", "left", "need"])
    recommendation = decision_tree.predict(new_purchase)[0]
    return recommendation

def get_cluster_plot_data():
    return student_data, cluster_names

def expense_chart(food, shopping, stationery, other):
    figure, axis = plt.subplots()
    axis.bar(
        ["Food", "Shopping", "Stationery", "Other"],
        [food, shopping, stationery, other]
    )
    axis.set_ylabel("Amount (Rs)")
    return figure

def confusion_table():
    return pd.DataFrame(model_confusion_matrix,
        index=["Actual Buy Now", "Actual Choose Cheaper", "Actual Postpone"],
        columns=["Predicted Buy Now", "Predicted Choose Cheaper", "Predicted Postpone"]
    )