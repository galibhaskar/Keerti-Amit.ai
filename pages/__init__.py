from pages.page_mapper import (
    settings_page,
    battle_mode_page,
    data_ingestion_page,
    practice_mode_page,
    logout_page,
    login_page,
)
import streamlit as st

def get_pages_for_role(role: str | None):
    if role in ["User", "Guest"]:
        account_pages = [settings_page, logout_page]
        page_dict = {
            "Data Ingestion": [data_ingestion_page],
            "Practice Mode": [practice_mode_page],
            "Battle Mode": [battle_mode_page],
        }
        return {"Account": account_pages} | page_dict
    else:
        return {"Account": [login_page]}