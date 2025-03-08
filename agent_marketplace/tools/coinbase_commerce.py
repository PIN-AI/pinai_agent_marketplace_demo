import streamlit as st
from textwrap import dedent

from agent_marketplace.schemas.agents import Message
from agent_marketplace.config import response_generator


def process_coinbase_payment(func_args: dict, message: Message) -> str:
    """
    A dummy function to process a Coinbase Commerce web3 payment using the provided JSON data
    """
    try:
        payment_json = message.metadata

        # Show payment details to the user
        payment_details = payment_json['paymentDetails']
        st.chat_message("user").write_stream(response_generator(dedent(f"""
            Payment Details:\n
            **Amount:** {payment_details['pricing']['local']['amount']} {payment_details['pricing']['local']['currency']}\n
            **Description:** {payment_details['metadata']['itemDescription']}\n
            **Client Name:** {payment_details['metadata']['name']}\n
            **Merchant:** {payment_details['organizationName']}\n
        """)))
        st.chat_message("user").write_stream(response_generator("⚠️ Please enter **YES** to confirm payment or **NO** to cancel in the :red[**TERMINAL**]."))
        
        # Prompt for user input in the terminal
        user_input = input("⚠️ \033[1;31mPlease enter YES to confirm payment or NO to cancel:\033[0m\n")

        # Process user input, if user confirms payment, return [PAYMENT_SUCCEEDED], otherwise return [PAYMENT_FAILED]
        if user_input.lower() == "yes":
            st.chat_message("user").write_stream(response_generator(":green[**[Payment confirmed]**]"))
            return "[PAYMENT_SUCCEEDED] The client has confirmed the order and payment is processed successfully"
        else:
            st.chat_message("user").write_stream(response_generator(":red[**[Payment cancelled]**]"))
            return "[PAYMENT_FAILED] The client has cancelled the payment. Please ends the conversation politely."
    
    except Exception as e:
        # If any error occurs, return [PAYMENT_FAILED]
        return f"[PAYMENT_FAILED] Error processing payment: {str(e)}. Please ends the conversation politely."
