export async function populateDeviceList(
  deviceSelect: HTMLSelectElement
): Promise<MediaDeviceInfo[]> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((d) => d.kind === "videoinput");
  deviceSelect.innerHTML = "";
  for (const d of videoInputs) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera (${d.deviceId.slice(0, 6)})`;
    deviceSelect.appendChild(opt);
  }
  return videoInputs;
}

export async function startStreamForDevice(
  video: HTMLVideoElement,
  currentStream: MediaStream | null,
  deviceId: string | null
): Promise<MediaStream> {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  const constraints: MediaStreamConstraints = {
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  return stream;
}
