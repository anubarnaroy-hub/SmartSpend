import streamlit as st
import matplotlib.pyplot as plt
from smartspend import (budget_report, spending_pattern, purchase_recommendation, 
expense_chart, model_accuracy,confusion_table,get_cluster_plot_data)

st.set_page_config(page_title="SmartSpend", page_icon="💰")
st.title("SmartSpend")
st.write("A simple student budget and purchase helper")

st.header("1. Enter your budget")
total_money = st.number_input("Total money (Rs)", min_value=1, value=5000, step=100)
spending_limit = st.number_input("Spending limit (Rs)", min_value=0, value=3500, step=100)

st.header("2. Enter your expenses")
food = st.number_input("Food (Rs)", min_value=0, value=500, step=50)
shopping = st.number_input("Shopping (Rs)", min_value=0, value=300, step=50)
stationery = st.number_input("Stationery / practical copies (Rs)", min_value=0, value=400, step=50)
other = st.number_input("Other expenses (Rs)", min_value=0, value=200, step=50)

if spending_limit > total_money:
    st.error("Your spending limit cannot be greater than your total money.")
else:
    report = budget_report(total_money, spending_limit, food, shopping, stationery, other)

    st.header("3. Budget summary")
    first_row = st.columns(3)
    first_row[0].metric("Total money", f"Rs {report['total']:,}")
    first_row[1].metric("Total spent", f"Rs {report['spent']:,}")
    first_row[2].metric("Money left overall", f"Rs {report['money_left']:,}")

    second_row = st.columns(3)
    second_row[0].metric("Left within spending limit", f"Rs {report['left']:,}")
    second_row[1].metric("Protected savings", f"Rs {report['savings']:,}")
    second_row[2].metric("Spending limit used", f"{report['budget_used_percent']:.1f}%")
    st.progress(min(report["budget_used_percent"], 100) / 100)

    if report["spent"] > report["total"]:
        st.error(f"You have spent Rs {report['spent'] - report['total']:,} more than all your available money.")
    elif report["left"] < 0:
        st.warning(f"You are over your spending limit by Rs {-report['left']:,}, but still have Rs {report['money_left']:,} overall.")
    else:
        st.success("Your expenses are within your spending limit.")

    st.caption("Protected savings = total money minus spending limit. This amount is planned to be kept aside.")

    st.header("4. Expense breakdown")
    st.pyplot(expense_chart(food, shopping, stationery, other))

    st.header("5. Spending pattern (K-Means clustering)")
    st.write("The graph groups example students by their Food and Shopping expenses.")
    st.write("Your marker is shown as a star (*). The model uses all four expense categories to assign your group.")
    samples, cluster_names = get_cluster_plot_data()
    figure, axis = plt.subplots()
    for group in sorted(samples["cluster"].unique()):
        group_data = samples[samples["cluster"] == group]
        axis.scatter(
            group_data["food"],
            group_data["shopping"],
            label=cluster_names[group],
            alpha=0.6
        )
    axis.scatter(food, shopping, marker="*", s=250, edgecolors="black", label="You")
    axis.set_xlabel("Food expenses (Rs)")
    axis.set_ylabel("Shopping expenses (Rs)")
    axis.legend()
    st.pyplot(figure)
    plt.close(figure)
    st.write("**Your spending group:**", spending_pattern(food, shopping, stationery, other))

    st.header("6. Purchase recommendation")
    product_name = st.text_input("What do you want to buy?")
    price = st.number_input("Product price (Rs)", min_value=0, value=500, step=50)
    needed = st.selectbox("Is this item needed?", ["Yes", "No"])

    parents_support = False
    if price > report["left"] and report["left"] >= 0:
        parent_answer = st.selectbox("Will your parents provide extra money if needed?", ["No", "Yes"])
        parents_support = parent_answer == "Yes"

    if st.button("Get recommendation"):
        if not product_name.strip():
            st.error("Please enter the product name.")
        elif price <= 0:
            st.error("Please enter a price greater than zero.")
        elif report["left"] < 0:
            st.warning("Review your expenses before planning another purchase.")
        else:
            advice = purchase_recommendation(price, report["left"], int(needed == "Yes"), parents_support)
            st.success(f"Recommendation for {product_name}: {advice}")

    with st.expander("About the AI model"):
        st.write(f"Decision Tree test accuracy on demo data: {model_accuracy * 100:.1f}%")
        st.write("Confusion matrix")
        st.dataframe(confusion_table())
        st.caption("The dataset is synthetic, so these results do not prove real-world accuracy.")
