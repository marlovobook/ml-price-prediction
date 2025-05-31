def display_metrics(total_return, annualized_return, max_drawdown):
    st.subheader("Model Performance Metrics")
    st.metric(label="Total Return", value=f"{total_return:.2f}%")
    st.metric(label="Annualized Return", value=f"{annualized_return:.2f}%")
    st.metric(label="Max Drawdown", value=f"{max_drawdown:.2f}%")