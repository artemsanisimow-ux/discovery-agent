"""
Discovery Agent — Production версия
=====================================
Установка:
    pip install langgraph langchain-anthropic langfuse langchain-core

Использование:
    export ANTHROPIC_API_KEY=sk-ant-...
    export LANGFUSE_PUBLIC_KEY=pk-lf-...   # опционально, для observability
    export LANGFUSE_SECRET_KEY=sk-lf-...   # опционально
    python discovery_agent_v3.py
"""

import os
import json
import sqlite3
from typing import TypedDict, Optional, Literal
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────
# LANGFUSE (опционально)
# ─────────────────────────────────────────────

try:
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
except ImportError:
    LANGFUSE_ENABLED = False
    langfuse_handler = None

# ─────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────

model = ChatAnthropic(model="claude-opus-4-5", max_tokens=4096)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class DiscoveryState(TypedDict):
    # Входные данные
    product_context: str
    raw_inputs: dict

    # Discovery
    insights: str
    data_sufficient: bool
    missing_data: str

    # Backlog
    backlog_draft: str
    backlog_final: str

    # Гипотезы
    hypotheses: str
    hypotheses_validated: bool

    # Управление процессом
    iteration: int
    current_step: str
    human_feedback: str
    needs_human: bool
    human_approved: bool
    done: bool

    # Audit
    audit: list
    session_id: str


# ─────────────────────────────────────────────
# AUDIT
# ─────────────────────────────────────────────

def log(state: DiscoveryState, step: str, data: dict) -> list:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "iteration": state.get("iteration", 0),
        "data": {k: str(v)[:300] for k, v in data.items()},
    }
    audit = state.get("audit", [])
    print(f"\n[{step.upper()}] итерация {entry['iteration']}")
    return audit + [entry]


# ─────────────────────────────────────────────
# ПРОМПТЫ
# ─────────────────────────────────────────────

RESEARCH_PROMPT = """Ты продакт-менеджер. Проанализируй данные и сформируй инсайты.

Контекст: {context}
Данные: {inputs}
Фидбек от предыдущей итерации: {feedback}

ВАЖНО: твой ответ должен быть ТОЛЬКО валидным JSON объектом. Никакого текста до или после. Никаких markdown блоков. Никаких ```json. Только сам JSON.

Формат ответа:
{{"insights": "структурированные инсайты по разделам", "data_sufficient": true, "missing_data": ""}}

Правила:
- data_sufficient = true если данных достаточно для формирования бэклога
- data_sufficient = false только если критически не хватает данных
- missing_data = пустая строка если data_sufficient = true"""

BACKLOG_PROMPT = """Сформируй черновик бэклога на основе инсайтов.

Контекст: {context}
Инсайты: {insights}
Гипотезы (если есть): {hypotheses}

ВАЖНО: твой ответ должен быть ТОЛЬКО валидным JSON объектом. Никакого текста до или после. Никаких markdown блоков. Никаких ```json. Только сам JSON.

Формат ответа:
{{"backlog": [{{"id": "B-001", "title": "название", "problem": "проблема", "hypothesis": "гипотеза", "priority": "P0", "effort": "M", "metric": "метрика", "risks": "риски"}}], "needs_human_review": false, "review_reason": ""}}"""

HYPOTHESIS_PROMPT = """Сформируй и оцени гипотезы для проверки.

Инсайты: {insights}
Бэклог: {backlog}

ВАЖНО: твой ответ должен быть ТОЛЬКО валидным JSON объектом. Никакого текста до или после. Никаких markdown блоков. Никаких ```json. Только сам JSON.

Формат ответа:
{{"hypotheses": [{{"id": "H-001", "statement": "формулировка", "confidence": "medium", "validation_method": "метод", "timeline": "2 недели"}}], "high_risk_items": ""}}"""


# ─────────────────────────────────────────────
# УЗЛЫ
# ─────────────────────────────────────────────

def research(state: DiscoveryState) -> DiscoveryState:
    """Исследование: анализируем данные, проверяем достаточность."""
    raw_str = "\n".join([
        f"[{k}]: {v}" for k, v in state["raw_inputs"].items() if v.strip()
    ])

    response = model.invoke(
        [
            SystemMessage(content=RESEARCH_PROMPT.format(
                context=state["product_context"],
                inputs=raw_str,
                feedback=state.get("human_feedback", "нет"),
            )),
            HumanMessage(content="Проведи анализ и верни JSON."),
        ],
        config={"callbacks": [langfuse_handler]} if LANGFUSE_ENABLED else {}
    )

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception:
        result = {
            "insights": response.content,
            "data_sufficient": True,
            "missing_data": ""
        }

    audit = log(state, "research", {
        "data_sufficient": result.get("data_sufficient"),
        "missing_data": result.get("missing_data", ""),
        "insights_preview": result.get("insights", "")[:200],
    })

    return {
        **state,
        "insights": result.get("insights", ""),
        "data_sufficient": result.get("data_sufficient", False),
        "missing_data": result.get("missing_data", ""),
        "current_step": "research",
        "audit": audit,
    }


def request_more_data(state: DiscoveryState) -> DiscoveryState:
    """
    Данных не хватает — останавливаемся и просим человека.
    После ответа — возвращаемся в research.
    """
    print("\n" + "="*60)
    print("⚠️  НУЖНЫ ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ")
    print("="*60)
    print(f"\nЧего не хватает:\n{state['missing_data']}")
    print("\nДобавь данные и нажми Enter (или 'skip' чтобы продолжить без них):")

    feedback = input("→ ").strip()

    audit = log(state, "request_more_data", {
        "missing_data": state["missing_data"],
        "human_response": feedback,
    })

    if feedback.lower() == "skip":
        return {
            **state,
            "data_sufficient": True,
            "human_feedback": "Пользователь решил продолжить без дополнительных данных",
            "audit": audit,
        }

    # Добавляем фидбек в raw_inputs
    updated_inputs = dict(state["raw_inputs"])
    updated_inputs["additional_data"] = feedback

    return {
        **state,
        "raw_inputs": updated_inputs,
        "human_feedback": feedback,
        "data_sufficient": False,
        "audit": audit,
    }


def form_backlog(state: DiscoveryState) -> DiscoveryState:
    """Формируем черновик бэклога из инсайтов."""
    response = model.invoke(
        [
            SystemMessage(content=BACKLOG_PROMPT.format(
                context=state["product_context"],
                insights=state["insights"],
                hypotheses=state.get("hypotheses", "ещё не сформированы"),
            )),
            HumanMessage(content="Сформируй бэклог и верни JSON."),
        ],
        config={"callbacks": [langfuse_handler]} if LANGFUSE_ENABLED else {}
    )

    try:
        result = json.loads(response.content)
    except Exception:
        result = {"backlog": [], "needs_human_review": True, "review_reason": "Ошибка парсинга"}

    needs_human = result.get("needs_human_review", False)

    audit = log(state, "form_backlog", {
        "items_count": len(result.get("backlog", [])),
        "needs_human": needs_human,
        "review_reason": result.get("review_reason", ""),
    })

    return {
        **state,
        "backlog_draft": json.dumps(result.get("backlog", []), ensure_ascii=False, indent=2),
        "needs_human": needs_human,
        "current_step": "backlog",
        "audit": audit,
    }


def validate_hypotheses(state: DiscoveryState) -> DiscoveryState:
    """Формируем и оцениваем гипотезы."""
    response = model.invoke(
        [
            SystemMessage(content=HYPOTHESIS_PROMPT.format(
                insights=state["insights"],
                backlog=state["backlog_draft"],
            )),
            HumanMessage(content="Сформируй гипотезы и верни JSON."),
        ],
        config={"callbacks": [langfuse_handler]} if LANGFUSE_ENABLED else {}
    )

    try:
        result = json.loads(response.content)
    except Exception:
        result = {"hypotheses": [], "high_risk_items": ""}

    audit = log(state, "validate_hypotheses", {
        "hypotheses_count": len(result.get("hypotheses", [])),
        "high_risk": result.get("high_risk_items", ""),
    })

    return {
        **state,
        "hypotheses": json.dumps(result.get("hypotheses", []), ensure_ascii=False, indent=2),
        "hypotheses_validated": True,
        "current_step": "hypotheses",
        "audit": audit,
    }


def human_checkpoint(state: DiscoveryState) -> DiscoveryState:
    """
    Человек проверяет бэклог и гипотезы.
    Может утвердить, отклонить или попросить изменения.
    """
    print("\n" + "="*60)
    print("✋ ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ ЧЕЛОВЕКА")
    print("="*60)

    print("\n📋 ЧЕРНОВИК БЭКЛОГА:")
    try:
        backlog = json.loads(state["backlog_draft"])
        for item in backlog:
            print(f"\n  [{item.get('id', '?')}] {item.get('title', '')}")
            print(f"  Проблема: {item.get('problem', '')}")
            print(f"  Приоритет: {item.get('priority', '')} | Усилия: {item.get('effort', '')}")
            print(f"  Метрика: {item.get('metric', '')}")
            print(f"  Риски: {item.get('risks', '')}")
    except Exception:
        print(state["backlog_draft"])

    print("\n🔬 ГИПОТЕЗЫ:")
    try:
        hyps = json.loads(state.get("hypotheses", "[]"))
        for h in hyps:
            print(f"\n  [{h.get('id', '?')}] {h.get('statement', '')}")
            print(f"  Уверенность: {h.get('confidence', '')} | Проверка: {h.get('timeline', '')}")
    except Exception:
        print(state.get("hypotheses", ""))

    print("\n" + "="*60)
    print("Варианты:")
    print("  y — утвердить, сохранить финальный отчёт")
    print("  n — отклонить, указать что изменить")
    print("  r — вернуться к исследованию с новыми данными")

    choice = input("\nТвой выбор (y/n/r): ").strip().lower()

    audit = log(state, "human_checkpoint", {"choice": choice})

    if choice == "y":
        return {
            **state,
            "human_approved": True,
            "needs_human": False,
            "audit": audit,
        }
    elif choice == "r":
        extra = input("Какие данные добавить? → ").strip()
        updated = dict(state["raw_inputs"])
        updated["human_addition"] = extra
        return {
            **state,
            "raw_inputs": updated,
            "human_feedback": f"Возврат к исследованию: {extra}",
            "human_approved": False,
            "needs_human": False,
            "hypotheses_validated": False,
            "iteration": state.get("iteration", 0) + 1,
            "audit": audit,
        }
    else:
        feedback = input("Что изменить в бэклоге? → ").strip()
        return {
            **state,
            "human_feedback": feedback,
            "human_approved": False,
            "needs_human": False,
            "hypotheses_validated": False,
            "iteration": state.get("iteration", 0) + 1,
            "audit": audit,
        }


def finalize(state: DiscoveryState) -> DiscoveryState:
    """Сохраняем финальный отчёт."""
    session_id = state["session_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"discovery_report_{session_id}_{timestamp}.md"

    try:
        backlog = json.loads(state["backlog_draft"])
        backlog_md = "\n\n".join([
            f"### [{item.get('id')}] {item.get('title')}\n"
            f"**Проблема:** {item.get('problem')}\n"
            f"**Гипотеза:** {item.get('hypothesis')}\n"
            f"**Приоритет:** {item.get('priority')} | **Усилия:** {item.get('effort')}\n"
            f"**Метрика:** {item.get('metric')}\n"
            f"**Риски:** {item.get('risks')}"
            for item in backlog
        ])
    except Exception:
        backlog_md = state["backlog_draft"]

    report = f"""# Discovery Report
Сессия: {session_id}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Итераций: {state.get('iteration', 0)}

---

## Инсайты

{state['insights']}

---

## Бэклог

{backlog_md}

---

## Гипотезы

{state.get('hypotheses', '')}

---

## Audit Trail

| Шаг | Итерация | Время |
|-----|----------|-------|
"""
    for entry in state.get("audit", []):
        report += f"| {entry['step']} | {entry['iteration']} | {entry['timestamp']} |\n"

    Path(filename).write_text(report, encoding="utf-8")

    audit_file = f"audit_{session_id}_{timestamp}.json"
    Path(audit_file).write_text(
        json.dumps(state.get("audit", []), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n✅ Готово!")
    print(f"📄 Отчёт: {filename}")
    print(f"📋 Audit: {audit_file}")

    audit = log(state, "finalize", {"filename": filename})

    return {**state, "done": True, "backlog_final": report, "audit": audit}


# ─────────────────────────────────────────────
# РОУТЕРЫ
# ─────────────────────────────────────────────

def after_research(state: DiscoveryState) -> Literal["request_more_data", "form_backlog"]:
    if not state.get("data_sufficient", False):
        return "request_more_data"
    return "form_backlog"


def after_more_data(state: DiscoveryState) -> Literal["research", "form_backlog"]:
    if state.get("data_sufficient", False):
        return "form_backlog"
    return "research"


def after_backlog(state: DiscoveryState) -> Literal["validate_hypotheses", "human_checkpoint"]:
    # Сначала всегда формируем гипотезы если их ещё нет
    if not state.get("hypotheses_validated", False):
        return "validate_hypotheses"
    # Только после гипотез идём к человеку
    return "human_checkpoint"


def after_human(state: DiscoveryState) -> Literal["finalize", "research", "form_backlog"]:
    if state.get("human_approved", False):
        return "finalize"
    feedback = state.get("human_feedback", "")
    if "возврат к исследованию" in feedback.lower():
        return "research"
    return "form_backlog"

def after_hypotheses(state: DiscoveryState) -> Literal["form_backlog"]:
    return "form_backlog"


def after_hypotheses(state: DiscoveryState) -> Literal["form_backlog"]:
    # После гипотез всегда обновляем бэклог
    return "form_backlog"


# ─────────────────────────────────────────────
# ГРАФ
# ─────────────────────────────────────────────

def build_graph(db_path: str = "discovery.db"):
    graph = StateGraph(DiscoveryState)

    graph.add_node("research", research)
    graph.add_node("request_more_data", request_more_data)
    graph.add_node("form_backlog", form_backlog)
    graph.add_node("validate_hypotheses", validate_hypotheses)
    graph.add_node("human_checkpoint", human_checkpoint)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("research")

    graph.add_conditional_edges("research", after_research, {
        "request_more_data": "request_more_data",
        "form_backlog": "form_backlog",
    })

    graph.add_conditional_edges("request_more_data", after_more_data, {
        "research": "research",
        "form_backlog": "form_backlog",
    })

    graph.add_conditional_edges("form_backlog", after_backlog, {
        "validate_hypotheses": "validate_hypotheses",
        "human_checkpoint": "human_checkpoint",
    })

    graph.add_conditional_edges("validate_hypotheses", after_hypotheses, {
        "form_backlog": "form_backlog",
    })

    graph.add_conditional_edges("human_checkpoint", after_human, {
        "finalize": "finalize",
        "research": "research",
        "form_backlog": "form_backlog",
    })

    graph.add_edge("finalize", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────

def run_discovery(
    product_context: str,
    interviews: str = "",
    analytics: str = "",
    reviews: str = "",
    call_recordings: str = "",
    stakeholder_requests: str = "",
    session_id: str = None,
    resume: bool = False,
):
    """
    resume=True — продолжить прерванную сессию по session_id.
    resume=False — начать новую сессию.
    """
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    app = build_graph()
    config = {"configurable": {"thread_id": sid}}

    if resume:
        print(f"\n▶️  Продолжаю сессию {sid}...")
        state = app.get_state(config)
        print(f"Текущий шаг: {state.values.get('current_step', 'неизвестен')}")
        final_state = app.invoke(None, config=config)
    else:
        print(f"\n🚀 Новая сессия: {sid}")
        initial = DiscoveryState(
            product_context=product_context,
            raw_inputs={
                "interviews": interviews,
                "analytics": analytics,
                "reviews": reviews,
                "call_recordings": call_recordings,
                "stakeholder_requests": stakeholder_requests,
            },
            insights="",
            data_sufficient=False,
            missing_data="",
            backlog_draft="",
            backlog_final="",
            hypotheses="",
            hypotheses_validated=False,
            iteration=0,
            current_step="start",
            human_feedback="",
            needs_human=False,
            human_approved=False,
            done=False,
            audit=[],
            session_id=sid,
        )
        final_state = app.invoke(initial, config=config)

    return final_state, sid


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    final_state, sid = run_discovery(
        product_context="""
        Продукт: мобильное приложение для личных финансов (iOS/Android).
        Пользователь: люди 25-35 лет, хотят контролировать расходы.
        Ключевая метрика: DAU/MAU (сейчас 18%, цель 35%).
        Фокус: почему пользователи перестают вносить транзакции после первой недели.
        """,
        interviews="""
        Пользователь 1: "Первые дни вносил всё, потом надоело — слишком много времени."
        Пользователь 2: "Забываю вносить расходы, потом накапливается и бросаю."
        Пользователь 3: "Хотел автоматическую синхронизацию с банком, но её нет."
        Пользователь 4: "Категории не подходят под мои расходы."
        """,
        analytics="""
        Day-1 retention: 71%, Day-7: 34%, Day-30: 12%
        Транзакций в день: день 1-3 = 4.2, день 4-7 = 1.8, день 8+ = 0.3
        Точка выхода: экран ручного ввода. Время на ввод: 47 секунд.
        """,
        reviews="""
        "Слишком долго вводить каждую покупку" — 23 отзыва (★★)
        "Нет синхронизации с банком" — 18 отзывов (★★)
        "Лучшее приложение для бюджета" — 9 отзывов (★★★★★)
        """,
        stakeholder_requests="""
        CEO: почему retention падает с 71% до 12% за месяц.
        Dev: готовы сделать интеграцию с банками, нужен приоритет.
        """
    )

    print(f"\n💾 Session ID для возобновления: {sid}")
    print("Чтобы продолжить прерванную сессию:")
    print(f'  run_discovery(session_id="{sid}", resume=True, product_context="")')
