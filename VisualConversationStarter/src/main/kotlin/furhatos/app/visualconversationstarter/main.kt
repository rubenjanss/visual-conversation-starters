package furhatos.app.visualconversationstarter

import furhatos.app.visualconversationstarter.flow.*
import furhatos.event.EventSystem
import furhatos.skills.Skill
import furhatos.flow.kotlin.*
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.zeromq.ZMQ

val objserv = "tcp://192.168.0.103:9999" //The TCP socket of the object server

val subSocket: ZMQ.Socket = getConnectedSocket(zmq.ZMQ.ZMQ_SUB, objserv) //Makes a socket of the object server

/**
 * Function that starts a thread which continuously polls the object server.
 * Based on what is in the message will either send:
 *  - EnterEvent, for objects coming into view.
 *  - LeaveEvent, for objects going out of view.
 *
 *  These events can be caught in the flow (Main), and be responded to.
 */
fun startListenThread() {
    GlobalScope.launch { // launch a new coroutine in background and continue
        logger.warn("LAUNCHING COROUTINE")
        subSocket.subscribe("")
        while (true) {
            val message = subSocket.recvStr()
            logger.warn("got: $message")
            EventSystem.send(EnterEvent(message))
        }
    }
}

class VisualConversationStarterSkill : Skill() {
    override fun start() {
        startListenThread()
        Flow().run(Idle)
    }
}

fun main(args: Array<String>) {
    Skill.main(args)
}
