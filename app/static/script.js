const form = document.getElementById("upload-form");
const fileInput = document.getElementById("video-file");
const statusText = document.getElementById("status");
const grid = document.getElementById("faces-grid");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    statusText.textContent = "Please choose a video file first.";
    return;
  }

  const video = fileInput.files[0];
  const body = new FormData();
  body.append("file", video);

  statusText.textContent = "Uploading and detecting faces...";
  grid.innerHTML = "";

  try {
    const response = await fetch("/detect", {
      method: "POST",
      body,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Detection failed");
    }

    const { faces } = await response.json();
    if (!faces.length) {
      statusText.textContent = "No faces found. Try another video.";
      return;
    }

    statusText.textContent = `Found ${faces.length} face crops.`;
    faces.forEach((face) => {
      const card = document.createElement("article");
      card.className = "face-card";

      const img = document.createElement("img");
      img.src = face.image;
      img.alt = `Face at ${face.timestamp}s`;

      const caption = document.createElement("p");
      caption.textContent = `${face.timestamp}s`;

      card.appendChild(img);
      card.appendChild(caption);
      grid.appendChild(card);
    });
  } catch (error) {
    console.error(error);
    statusText.textContent = error.message;
  }
});
