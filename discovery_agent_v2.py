"""
Discovery Agent — с валидацией каждого шага и audit log
=========================================================
Установка:
    pip install langgraph langchain-anthropic python-dotenv

Использование:
    export ANTHROPIC_API_KEY=your_key
    python discovery_agent_v2.py
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from i18n import t, get_language, get_language_instruction
load_dotenv()
from datetime import datetime
import json

# ─────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────

model = ChatAnthropic(model="claude-opus-4-5", max_tokens=4096)

# ─────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────

class AuditLog:
    """Записывает что именно и в каких формулировках ушло в контекст."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.entries = []

    def record(self, step: str, data: dict, approved: bool, note: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step": step,
            "approved": approved,
            "note": note,
            "data": data,
        }
        self.entries.append(entry)
        self._print_entry(entry)

    def _print_entry(self, entry: dict):
        from i18n import t, LANGUAGE
        if LANGUAGE == "en":
            status = "✅ RECORDED" if entry["approved"] else "❌ REJECTED"
            step_label = "Step"
            reason_label = "Reason"
            context_label = "Data sent to context"
        else:
            status = "✅ ЗАПИСАНО" if entry["approved"] else "❌ ОТКЛОНЕНО"
            step_label = "Шаг"
            reason_label = "Причина"
            context_label = "Данные переданы в контекст"
        print(f"\n{'─'*60}")
        print(f"[AUDIT] {status} | {step_label}: {entry['step']} | {entry['timestamp']}")
        if entry["note"]:
            print(f"[AUDIT] {reason_label}: {entry['note']}")
        print(f"[AUDIT] {context_label}:")
        for key, value in entry["data"].items():
            preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            print(f"  • {key}: {preview}")
        print(f"{'─'*60}")

    def save(self, path: str = "discovery_audit.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        from i18n import LANGUAGE
        msg = "Audit log saved" if LANGUAGE == "en" else "Audit log сохранён"
        print(f"\n📋 {msg}: {path}")

    def summary(self) -> str:
        from i18n import LANGUAGE
        approved = [e for e in self.entries if e["approved"]]
        rejected = [e for e in self.entries if not e["approved"]]
        if LANGUAGE == "en":
            return (
                f"Session: {self.session_id}\n"
                f"Total steps: {len(self.entries)}\n"
                f"Recorded to context: {len(approved)}\n"
                f"Rejected: {len(rejected)}\n"
            )
        return (
            f"Сессия: {self.session_id}\n"
            f"Всего шагов: {len(self.entries)}\n"
            f"Записано в контекст: {len(approved)}\n"
            f"Отклонено: {len(rejected)}\n"
        )


# ─────────────────────────────────────────────
# ВАЛИДАТОР
# ─────────────────────────────────────────────

class StepValidator:
    """
    Валидирует данные перед передачей в контекст.
    Можно настроить правила под свой процесс.
    """

    @staticmethod
    def _rules():
        from i18n import LANGUAGE
        if LANGUAGE == "en":
            return {
                "insights": [
                    ("min_length", 200, "Insights too short — likely incomplete"),
                    ("has_section", "##", "No section structure — document unstructured"),
                    ("has_section_any", ["Hypothes", "Hypothesis", "Гипотез"], "No hypotheses section"),
                ],
                "critique": [
                    ("min_length", 100, "Critique too short — likely superficial"),
                ],
                "final": [
                    ("min_length", 300, "Final document too short"),
                    ("has_section", "##", "No section structure"),
                    ("no_placeholder", "[fill]", "Document contains unfilled fields"),
                    ("no_placeholder", "TODO", "Document contains TODOs"),
                ],
            }
        else:
            return {
                "insights": [
                    ("min_length", 200, "Инсайты слишком короткие — вероятно неполные"),
                    ("has_section", "##", "Нет структуры разделов — документ неструктурирован"),
                    ("has_section_any", ["Гипотез", "Hypothes"], "Нет раздела с гипотезами"),
                ],
                "critique": [
                    ("min_length", 100, "Критика слишком короткая — вероятно поверхностная"),
                ],
                "final": [
                    ("min_length", 300, "Финальный документ слишком короткий"),
                    ("has_section", "##", "Нет структуры разделов"),
                    ("no_placeholder", "[заполнить]", "Документ содержит незаполненные поля"),
                    ("no_placeholder", "TODO", "Документ содержит TODO"),
                ],
            }

    RULES = {}

    def validate(self, step_type: str, content: str) -> tuple[bool, list[str]]:
        """
        Возвращает (валидно, список ошибок).
        """
        rules = self._rules().get(step_type, [])
        errors = []

        for rule in rules:
            rule_type = rule[0]

            if rule_type == "min_length":
                _, min_len, msg = rule
                if len(content) < min_len:
                    errors.append(msg)

            elif rule_type == "has_section":
                _, keyword, msg = rule
                if keyword.lower() not in content.lower():
                    errors.append(msg)

            elif rule_type == "has_section_any":
                _, keywords, msg = rule
                if not any(kw.lower() in content.lower() for kw in keywords):
                    errors.append(msg)

            elif rule_type == "no_placeholder":
                _, placeholder, msg = rule
                if placeholder in content:
                    errors.append(msg)

        return len(errors) == 0, errors


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class DiscoveryState(TypedDict):
    raw_inputs: dict
    product_context: str

    draft_insights: str
    critique: str
    reflection_count: int

    final_insights: str
    human_approved: bool

    # Аудит
    audit_log: list          # Сериализованный лог (для хранения в State)
    context_snapshot: dict   # Что реально ушло в контекст на каждом шаге
    validation_errors: list  # Ошибки последней валидации


# ─────────────────────────────────────────────
# ГЛОБАЛЬНЫЕ ОБЪЕКТЫ СЕССИИ
# ─────────────────────────────────────────────

_audit: Optional[AuditLog] = None
_validator = StepValidator()


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        _audit = AuditLog(session_id=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return _audit


# ─────────────────────────────────────────────
# ПРОМПТЫ
# ─────────────────────────────────────────────

SYNTHESIZER_PROMPT = """{lang}Ты опытный продакт-менеджер. Синтезируй сырые данные в структурированные инсайты.

Контекст продукта:
{product_context}

Сырые данные:
{raw_inputs}

Создай документ со следующими разделами:

## Ключевые проблемы пользователей
(топ 3-5 проблем с частотой упоминания)

## Паттерны поведения
(что пользователи делают, а не говорят)

## Сигналы из данных
(аналитика, отзывы, метрики)

## Гипотезы для проверки
(предположения, которые требуют проверки)

## Противоречия и пробелы
(где данные расходятся или чего не хватает)

Каждый пункт — конкретный и actionable."""

CRITIC_PROMPT = """{lang}Ты строгий критик. Найди слабые места в анализе.

Контекст: {product_context}

Документ:
{draft_insights}

Проверь:
1. ДОКАЗАТЕЛЬНОСТЬ — есть ли конкретные примеры?
2. ПРОТИВОРЕЧИЯ — где данные расходятся с выводами?
3. ПРОБЕЛЫ — что упущено?
4. ПРЕДВЗЯТОСТЬ — есть ли поспешные выводы?
5. ПРИОРИТЕТЫ — правильно ли расставлены?

Если всё хорошо — напиши "APPROVED" в начале."""

REFINER_PROMPT = """{lang}Улучши документ на основе критики.

Контекст: {product_context}
Документ: {draft_insights}
Критика: {critique}

Исправь все проблемы. Сохрани структуру."""


# ─────────────────────────────────────────────
# УЗЛЫ С ВАЛИДАЦИЕЙ
# ─────────────────────────────────────────────

def synthesize(state: DiscoveryState) -> DiscoveryState:
    print("\n" + t("synthesizing"))

    raw_str = "\n\n".join([
        f"### {source.upper()}\n{content}"
        for source, content in state["raw_inputs"].items()
        if content.strip()
    ])

    response = model.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT.format(lang=get_language_instruction(),
            product_context=state["product_context"],
            raw_inputs=raw_str
        )),
        HumanMessage(content=t("prompt_synthesize"))
    ])

    content = response.content

    # Валидация
    valid, errors = _validator.validate("insights", content)
    audit = get_audit()

    if valid:
        audit.record(
            step="synthesize",
            data={
                "draft_insights": content,
                "sources_used": list(state["raw_inputs"].keys()),
                "char_count": len(content),
            },
            approved=True,
            note="Passed validation" if __import__("i18n").LANGUAGE == "en" else "Прошло валидацию"
        )
    else:
        audit.record(
            step="synthesize",
            data={"draft_insights": content, "errors": errors},
            approved=False,
            note=f"Ошибки валидации: {'; '.join(errors)}"
        )
        print(f"⚠️  {errors}")

    return {
        **state,
        "draft_insights": content,
        "reflection_count": 0,
        "validation_errors": errors,
        "audit_log": audit.entries,
        "context_snapshot": {
            **state.get("context_snapshot", {}),
            "after_synthesize": {
                "content_length": len(content),
                "valid": valid,
                "errors": errors,
            }
        }
    }


def critique(state: DiscoveryState) -> DiscoveryState:
    print(t("critiquing"))

    response = model.invoke([
        SystemMessage(content=CRITIC_PROMPT.format(lang=get_language_instruction(),
            product_context=state["product_context"],
            draft_insights=state["draft_insights"]
        )),
        HumanMessage(content="Проверь.")
    ])

    content = response.content
    valid, errors = _validator.validate("critique", content)
    audit = get_audit()

    audit.record(
        step="critique",
        data={
            "critique": content,
            "approved_by_critic": "APPROVED" in content,
            "iteration": state.get("reflection_count", 0),
        },
        approved=valid,
        note=("APPROVED" if "APPROVED" in content else ("Issues found" if __import__("i18n").LANGUAGE == "en" else "Найдены замечания"))
    )

    return {
        **state,
        "critique": content,
        "audit_log": audit.entries,
        "context_snapshot": {
            **state.get("context_snapshot", {}),
            f"after_critique_{state.get('reflection_count', 0)}": {
                "critic_approved": "APPROVED" in content,
                "critique_length": len(content),
            }
        }
    }


def should_refine(state: DiscoveryState) -> str:
    critique_text = state.get("critique", "")
    count = state.get("reflection_count", 0)
    if "APPROVED" in critique_text or count >= 2:
        return "finalize"
    return "refine"


def refine(state: DiscoveryState) -> DiscoveryState:
    count = state.get("reflection_count", 0)
    print(t("refining", n=count + 1))

    response = model.invoke([
        SystemMessage(content=REFINER_PROMPT.format(lang=get_language_instruction(),
            product_context=state["product_context"],
            draft_insights=state["draft_insights"],
            critique=state["critique"]
        )),
        HumanMessage(content="Улучши.")
    ])

    content = response.content
    valid, errors = _validator.validate("insights", content)
    audit = get_audit()

    audit.record(
        step=f"refine_{count + 1}",
        data={
            "refined_insights": content,
            "based_on_critique": state["critique"][:300] + "...",
            "iteration": count + 1,
            "char_count": len(content),
        },
        approved=valid,
        note=f"Iteration {count + 1}" + (f" | Errors: {errors}" if errors else "") if __import__("i18n").LANGUAGE == "en" else f"Итерация {count + 1}" + (f" | Ошибки: {errors}" if errors else "")
    )

    return {
        **state,
        "draft_insights": content,
        "reflection_count": count + 1,
        "validation_errors": errors,
        "audit_log": audit.entries,
    }


def finalize(state: DiscoveryState) -> DiscoveryState:
    print(t("finalizing"))

    final = f"""# Discovery Report
Сессия: {get_audit().session_id}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'─' * 50}

{state['draft_insights']}

{'─' * 50}
Итераций рефлексии: {state.get('reflection_count', 0)}
"""

    valid, errors = _validator.validate("final", final)
    audit = get_audit()

    audit.record(
        step="finalize",
        data={
            "final_document": final,
            "total_iterations": state.get("reflection_count", 0),
            "char_count": len(final),
            "validation_passed": valid,
        },
        approved=valid,
        note=("Ready for review" if valid else f"Issues: {errors}") if __import__("i18n").LANGUAGE == "en" else ("Готов к проверке человеком" if valid else f"Проблемы: {errors}")
    )

    return {
        **state,
        "final_insights": final,
        "human_approved": False,
        "audit_log": audit.entries,
    }


def human_review(state: DiscoveryState) -> DiscoveryState:
    audit = get_audit()

    print("\n" + "=" * 60)
    print(t("finalizing"))
    print("=" * 60)
    print(state["final_insights"])

    # Показываем что реально ушло в контекст
    print("\n" + "=" * 60)
    print("🔍 ЧТО УШЛО В КОНТЕКСТ (snapshot):")
    print("=" * 60)
    for step, data in state.get("context_snapshot", {}).items():
        print(f"\n  [{step}]")
        for k, v in data.items():
            print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print(f"📊 AUDIT SUMMARY:\n{audit.summary()}")

    answer = input("\n" + t("approve_prompt")).strip().lower()

    if answer == "y":
        audit.record(
            step="human_review",
            data={"decision": "approved", "final_insights": state["final_insights"]},
            approved=True,
            note=t("human_approved_note")
        )
        audit.save()
        return {**state, "human_approved": True, "audit_log": audit.entries}
    else:
        feedback = input(t("feedback_prompt"))
        audit.record(
            step="human_review",
            data={"decision": "rejected", "feedback": feedback},
            approved=False,
            note=f"Отклонено PM: {feedback}"
        )
        return {
            **state,
            "critique": f"Фидбек от PM: {feedback}",
            "human_approved": False,
            "audit_log": audit.entries,
        }


def after_review(state: DiscoveryState) -> str:
    return "done" if state["human_approved"] else "refine"


# ─────────────────────────────────────────────
# ГРАФ
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(DiscoveryState)

    graph.add_node("synthesize", synthesize)
    graph.add_node("critique", critique)
    graph.add_node("refine", refine)
    graph.add_node("finalize", finalize)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges("critique", should_refine, {
        "refine": "refine",
        "finalize": "finalize"
    })
    graph.add_edge("refine", "critique")
    graph.add_edge("finalize", "human_review")
    graph.add_conditional_edges("human_review", after_review, {
        "refine": "refine",
        "done": END
    })

    return graph.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────

def run_discovery(
    product_context: str,
    interviews: str = "",
    call_recordings: str = "",
    reviews: str = "",
    analytics: str = "",
    stakeholder_requests: str = "",
    thread_id: str = None,
):
    global _audit
    _audit = AuditLog(session_id=thread_id or datetime.now().strftime("%Y%m%d_%H%M%S"))

    app = build_graph()

    initial_state = DiscoveryState(
        raw_inputs={
            "interviews": interviews,
            "call_recordings": call_recordings,
            "reviews": reviews,
            "analytics": analytics,
            "stakeholder_requests": stakeholder_requests,
        },
        product_context=product_context,
        draft_insights="",
        critique="",
        reflection_count=0,
        final_insights="",
        human_approved=False,
        audit_log=[],
        context_snapshot={},
        validation_errors=[],
    )

    config = {"configurable": {"thread_id": _audit.session_id}}
    final_state = app.invoke(initial_state, config=config)

    # Сохраняем финальный отчёт
    with open("discovery_report.md", "w", encoding="utf-8") as f:
        f.write(final_state["final_insights"])
    print("\n" + t("report_saved") + ": discovery_report.md")

    return final_state


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_discovery(
        product_context="""
        B2B SaaS для HR-команд. Пользователь: HR-менеджер, 50-500 чел.
        Ключевая метрика: activation rate (28%, цель 45%).
        Фокус: онбординг и первая неделя использования.
        """,
        interviews="""
        Пользователь 1: "Первые 2 дня не понял что делать. Нет понятного старта."
        Пользователь 2: "Интеграция с 1С заняла 3 недели. Документация устаревшая."
        Пользователь 3: "Продукт нравится, но команда не хочет переходить."
        """,
        analytics="""
        Онбординг: step1=82%, step2=61%, step3=34%, step4=19%
        Day-7 retention: 34%. Median time-to-value: 8 дней.
        Самый частый первый action: импорт сотрудников (67%).
        Точка выхода: экран настройки интеграций.
        """,
        reviews="""
        "Сложная настройка" — 14 отзывов (★★)
        "Хорошо работает после настройки" — 8 отзывов (★★★★)
        "Поддержка не отвечает" — 6 отзывов (★)
        """,
        stakeholder_requests="""
        Sales: клиенты просят API для Bitrix24 (5 запросов)
        Support: 40% тикетов — настройка интеграций
        CEO: почему activation rate не растёт несмотря на фичи?
        """
    )
