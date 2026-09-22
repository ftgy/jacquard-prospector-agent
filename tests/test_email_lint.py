"""Deterministic playbook checks on drafted emails (no network)."""

from prospector.agent import SELF_INTRO, SENDER_NAME, SIGNATURE_LINKS
from prospector.email_lint import lint_email

SUBJECT = "¿Cuánto tarda un CV en estar listo para enviar?"


def body(observation="Vi que en vuestra web decís que vuestros reclutadores son "
                     "expertos en talento. Imagino que buena parte de la semana se "
                     "va en reformatear CVs. ¿Cuánto tiempo pierde hoy el equipo en "
                     "eso?",
         ask="¿Os cuadra agendar una llamada de 20 minutos para comentarlo? Si ya lo "
             "tenéis cubierto, quedo a vuestra disposición para otra ocasión.",
         intro=SELF_INTRO["spanish"], signature=f"{SENDER_NAME}\n{SIGNATURE_LINKS}"):
    return (f"Buenas,\n\n{observation}\n\n{intro} Para un caso como el vuestro "
            "montaría un sistema que se encargue de ese flujo concreto; esto suele "
            f"resolverse en un par de semanas.\n\n{ask}\n\nUn saludo,\n{signature}")


def rules(issues):
    return {i["rule"] for i in issues}


def test_clean_email_passes():
    assert lint_email(SUBJECT, body()) == []


def test_fixed_lines_survive_rewrapping():
    rewrapped = body().replace(SELF_INTRO["spanish"],
                               SELF_INTRO["spanish"].replace(" y me ", "\ny me "))
    assert lint_email(SUBJECT, rewrapped) == []


def test_paraphrased_intro_is_flagged():
    b = body(intro="Soy Francisco, ingeniero que automatiza procesos con IA.")
    assert "self-intro" in rules(lint_email(SUBJECT, b))


def test_reworded_ask_is_flagged():
    b = body(ask="¿Tendría sentido una llamada de 20 minutos?")
    found = rules(lint_email(SUBJECT, b))
    assert {"ask-opener", "banned"} <= found


def test_usted_and_pequeno_are_flagged():
    b = body(observation="Imagino que usted pierde horas. Montaría un agente pequeño.")
    details = " ".join(i["detail"] for i in lint_email(SUBJECT, b))
    assert "usted" in details and "pequeño" in details


def test_missing_signature_is_flagged():
    b = body(signature=SENDER_NAME)
    assert "signature" in rules(lint_email(SUBJECT, b))


def test_url_in_body_is_flagged_but_not_signature():
    assert "url-in-body" not in rules(lint_email(SUBJECT, body()))
    b = body(observation="Mirad feina.dev para ver ejemplos.")
    assert "url-in-body" in rules(lint_email(SUBJECT, b))


def test_long_email_and_long_sentence_are_flagged():
    long_sentence = " ".join(["palabra"] * 60) + "."
    b = body(observation=" ".join([long_sentence] * 5))
    assert {"length", "long-sentence"} <= rules(lint_email(SUBJECT, b))


def test_subject_rules():
    assert "subject" in rules(lint_email("Automatizar la gestión de CVs", body()))
    assert "subject" in rules(lint_email("Agentes de IA para vuestro equipo", body()))


def test_followup_skips_intro_ask_and_subject_rules():
    followup = ("Buenas,\n\nOs escribí hace unos días sobre los CVs. Imagino que "
                "sigue siendo una tarea que se come horas. Si queréis, lo vemos en 20 "
                "minutos; si no es el momento, lo entiendo.\n\nUn saludo,\n"
                f"{SENDER_NAME}\n{SIGNATURE_LINKS}")
    assert lint_email("Re: Automatizar la IA de CVs", followup, followup=True) == []
    assert {"self-intro", "ask-opener", "subject"} <= rules(
        lint_email("Re: Automatizar la IA de CVs", followup))


def test_followup_has_a_tighter_word_ceiling():
    b = body(observation=" ".join(["palabra."] * 60))
    assert "length" not in rules(lint_email(SUBJECT, b))
    assert "length" in rules(lint_email(SUBJECT, b, followup=True))
