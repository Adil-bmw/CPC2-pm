import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Smart Matching AI", layout="wide")
st.title("Умный мэтчинг студентов и научных руководителей 🎓")

st.sidebar.header("Настройки")
api_key = st.sidebar.text_input("Введи Gemini API Key:", type="password")

st.markdown("---")

# Зона 1: Редактирование агентов и задач
st.header("Зона 1: Конфигурация агентов")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Агент 1: Аналитик заявок")
    agent1_role = st.text_input("Role 1", value="Аналитик студенческих проектов")
    agent1_goal = st.text_input("Goal 1", value="Извлечь ключевые навыки, интересы и тему проекта из заявки студента.")
    agent1_backstory = st.text_area("Backstory 1",
                                    value="Ты эксперт в анализе образовательных запросов и технологических трендов. Ты понимаешь суть студенческих идей.")
    task1_desc = st.text_area("Описание задачи 1",
                              value="Проанализируй следующую заявку студента: {student_application}. Составь краткое резюме его интересов и требуемого стека технологий.")

with col2:
    st.subheader("Агент 2: Специалист по мэтчингу")
    agent2_role = st.text_input("Role 2", value="Академический координатор")
    agent2_goal = st.text_input("Goal 2",
                                value="Найти лучшего научного руководителя для студента на основе базы преподавателей.")
    agent2_backstory = st.text_area("Backstory 2",
                                    value="Ты знаешь профили всех преподавателей. Твоя цель — создать идеальный союз студента и наставника.")
    task2_desc = st.text_area("Описание задачи 2",
                              value="Используя резюме студента от первого агента и список преподавателей: {professors_json}, выбери наиболее подходящего руководителя. Обоснуй выбор.")

st.markdown("---")

# Зона 2: Ввод переменных данных
st.header("Зона 2: Ввод данных")

student_app_input = st.text_area("Текст заявки студента",
                                 value="Хочу создать веб-приложение для генерации музыки с помощью нейросетей. Знаю Python, хочу изучить PyTorch и машинное обучение.")

professors_json_input = st.text_area("JSON-профили преподавателей",
                                     value='[{"name": "Иванов И.И.", "expertise": "Компьютерное зрение, OpenCV"}, {"name": "Петров А.С.", "expertise": "Глубокое обучение, PyTorch, генеративные сети"}]')

st.markdown("---")

# Зона 3: Запуск и визуализация
st.header("Зона 3: Запуск экипажа")

if st.button("Сохранить и запустить"):
    if not api_key:
        st.error("Пожалуйста, введи Gemini API Key в боковой панели!")
    else:
        try:
            # Установка ключа в переменные окружения
            os.environ["GOOGLE_API_KEY"] = api_key


            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                verbose=True,
                temperature=0.5,
                google_api_key=api_key
            )

            # Инициализация агентов
            researcher = Agent(
                role=agent1_role,
                goal=agent1_goal,
                backstory=agent1_backstory,
                llm=llm,
                verbose=True
            )

            matcher = Agent(
                role=agent2_role,
                goal=agent2_goal,
                backstory=agent2_backstory,
                llm=llm,
                verbose=True
            )

            # Инициализация задач
            task1 = Task(
                description=task1_desc,
                expected_output="Структурированное резюме студента (интересы, стек).",
                agent=researcher
            )

            task2 = Task(
                description=task2_desc,
                expected_output="Имя выбранного преподавателя и детальное обоснование выбора.",
                agent=matcher
            )


            crew = Crew(
                agents=[researcher, matcher],
                tasks=[task1, task2],
                process=Process.sequential
            )

            with st.spinner("Агенты анализируют данные..."):
                result = crew.kickoff(inputs={
                    'student_application': student_app_input,
                    'professors_json': professors_json_input
                })

                st.success("Готово!")
                st.markdown("### Результат мэтчинга:")
                st.write(result.raw)

        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")