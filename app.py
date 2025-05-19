            MACD: {df['MACD'].iloc[-1]:.2f}
            SMA20: {df['SMA20'].iloc[-1]:.2f}
            News Sentiment: {avg_compound:.2f} ({verdict})

            You are a financial analyst. Based on the technical indicators and the news sentiment,
            provide a short and clear analysis of the stock situation in English.
            """

            try:
                client = OpenAI(api_key=OPENAI_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.success("🔍 AI-generated Analysis:")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.warning(f"⚠️ Error generating analysis with OpenAI: {str(e)}")
        else:
            st.warning("🔑 Please set your OPENAI_API_KEY in Streamlit Secrets.")


# ─── FINAL DECISION INDICATOR ─────────────────────────────────────
st.markdown("---")
st.header("📍 Final Recommendation")



@@ -209,4 +243,3 @@ try:

except Exception as e:
    st.warning(f"⚠️ Couldn't generate final recommendation: {str(e)}")
