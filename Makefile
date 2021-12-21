SHELL = /bin/sh
CURRENT_UID := $(shell id -u)

run:
	streamlit run main.py