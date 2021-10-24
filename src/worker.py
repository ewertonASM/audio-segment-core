import base64
import json
import logging.config
import os


from process.segmentAudio import silenceRemoval
from util import configreader
from queues import queuewrapper


class Worker:
    
    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)

        self.consumer = queuewrapper.QueueConsumer()
        self.publisher = queuewrapper.QueuePublisher()     

    def __reply_message(self, route, message):
        self.__logger.info("Sending response to request.")

        if id is None:
            self.__logger.error("The request don't have correlation_id.")

        if route is None:
            self.__logger.error("The request don't have reply_to route.")
        else:
            self.publisher.publish_to_queue(route, message)


    def __callback(self, _ch, _method, _properties, body):

        tmp_dir = "tmp/"
        tmp_audio_dir = f'{tmp_dir}/tmp.wav'
        payload = body.decode("utf-8")
        message = json.loads(payload)


        audio = base64.b64decode(message["audio"])

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        with open(tmp_audio_dir, "wb") as f:
            f.write(audio)

        self.__logger.info("Segmenting audio...")
        snippets = silenceRemoval(tmp_audio_dir)
        os.remove(tmp_audio_dir)

        for index, snippet in enumerate(snippets):

            with open(snippet, "rb") as f:
                audio_file = f.read()

            msg = {
                "videoId": message["videoId"],
                "interval": snippet.split('_')[1],
                "snippet": f'{index+1}/{len(snippets)}',
                "audio": base64.b64encode(audio_file).decode('utf-8')
            }

            payload = json.dumps(msg)

            self.__reply_message(
                route=workercfg.get("TranscriptionQueue"),
                message=payload)

            os.remove(snippet)

    def start(self, queue):
        self.__logger.debug("Starting queue consumer.")
        self.consumer.consume_from_queue(queue, self.__callback)

    def stop(self):
        self.__logger.debug("Stopping queue consumer.")
        self.consumer.close_connection()
        self.__logger.debug("Stopping queue publisher.")
        self.publisher.close_connection()


if __name__ == "__main__":
    logging.config.fileConfig(os.environ.get("LOGGER_CONFIG_FILE", ""))
    logger = logging.getLogger(__name__)

    workercfg = configreader.load_configs("Worker")

    if not workercfg:
        raise SystemExit(1)

    try:
        logger.info("Creating Audio Processor Worker.")
        worker = Worker()
        logger.info("Starting Audio Processor Worker.")
        worker.start(workercfg.get("AudioExtractQueue"))

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: stopping Transcription Worker.")

    except Exception:
        logger.exception("Unexpected error has occured in Transcription Worker.")

    finally:
        worker.stop()
        raise SystemExit(1)
