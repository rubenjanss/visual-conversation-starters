package furhatos.app.visualconversationstarter.flow

import furhatos.flow.kotlin.*

val Start : State = state(Interaction) {

    onEntry {
        furhat.cameraFeed.enable()
        logger.info(furhat.cameraFeed.isOpen())
    }
}
