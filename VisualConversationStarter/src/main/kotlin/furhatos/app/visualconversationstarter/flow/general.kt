package furhatos.app.visualconversationstarter.flow

import furhatos.event.Event
import furhatos.flow.kotlin.*
import furhatos.flow.kotlin.voice.Voice
import furhatos.util.*

val logger = CommonUtils.getRootLogger()

class EnterEvent(val msg: String): Event()

val Idle: State = state {

    init {
        logger.info("Initializing Idle state")
        furhat.voice = Voice(language = Language.ENGLISH_US, gender = Gender.MALE, volume="soft")
        if (users.count > 0) {
            furhat.attend(users.random)
            goto(Start)
        }
    }

    onEntry {
        furhat.attendNobody()
    }

    onUserEnter {
        furhat.attend(it)
        goto(Start)
    }

}

val Interaction: State = state {

    onUserLeave(instant = true) {
        if (users.count > 0) {
            if (it == users.current) {
                furhat.attend(users.other)
                goto(Start)
            } else {
                furhat.glance(it)
            }
        } else {
            furhat.cameraFeed.disable()
            logger.info(furhat.cameraFeed.port())
            goto(Idle)
        }
    }

    onUserEnter(instant = true) {
        furhat.glance(it)
    }

    onEvent<EnterEvent> {// Objects that enter the view
        furhat.say("Hi there!")
        furhat.say(it.msg)
    }

}