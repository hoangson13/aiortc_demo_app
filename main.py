import argparse
import asyncio
import json
import logging
import os
import ssl
import traceback
import uuid

import cv2
from aiohttp import web
import mediapipe as mp

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

from utils import validate

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


class FaceLivelinessProcessor:
    def __init__(self):
        self.__tracks = {}
        self.channel = None
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def addTrack(self, track):
        if track not in self.__tracks:
            self.__tracks[track] = None

    async def start(self):
        for track, task in self.__tracks.items():
            if task is None:
                self.__tracks[track] = asyncio.ensure_future(self.consume(track))

    async def stop(self):
        for task in self.__tracks.values():
            if task is not None:
                task.cancel()
        self.__tracks = {}
        self.face_mesh.close()

    def detect_img(self, image):
        # Convert the color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance
        image.flags.writeable = False
        # Get the result
        results = self.face_mesh.process(image)
        # To improve performance
        image.flags.writeable = True
        return results

    async def consume(self, track):
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                img_h, img_w, img_c = img.shape

                results = self.detect_img(img)

                if self.channel is not None:
                    if results.multi_face_landmarks:
                        if len(results.multi_face_landmarks) == 1:
                            box, direction, blink, smile = validate(results.multi_face_landmarks[0], img_h, img_w)
                            mess = {
                                "num_face": 1,
                                "direction": direction,
                                "blink": blink,
                                "smile": smile,
                                "box": box
                            }
                        else:
                            mess = {
                                "num_face": len(results.multi_face_landmarks),
                            }
                    else:
                        mess = {"num_face": 0}
                    # print(mess)
                    self.channel.send(json.dumps(mess))
                    await self.channel._RTCDataChannel__transport._data_channel_flush()
                    await self.channel._RTCDataChannel__transport._transmit()

            except Exception as e:
                traceback.print_exc()
                return


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PeerConnection({uuid.uuid4()})"
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info(f"Created for {request.remote}")

    # prepare local media
    processor = FaceLivelinessProcessor()

    @pc.on("datachannel")
    def on_datachannel(channel):
        processor.channel = channel
        log_info("Mount data channel")

        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            processor.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await processor.stop()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # handle offer
    await pc.setRemoteDescription(offer)
    await processor.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
