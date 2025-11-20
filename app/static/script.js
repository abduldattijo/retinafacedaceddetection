const form = document.getElementById("upload-form");
const fileInput = document.getElementById("video-file");
const statusText = document.getElementById("status");
const grid = document.getElementById("faces-grid");
const thresholdSlider = document.getElementById("match-threshold");
const thresholdValue = document.getElementById("threshold-value");

if (thresholdSlider && thresholdValue) {
  const updateDisplay = () => {
    thresholdValue.textContent = Number(thresholdSlider.value).toFixed(2);
  };
  thresholdSlider.addEventListener("input", updateDisplay);
  updateDisplay();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    statusText.textContent = "Please choose a video file first.";
    return;
  }

  const mode = document.querySelector('input[name="mode"]:checked')?.value || "search";
  const threshold = thresholdSlider ? Number(thresholdSlider.value || 0.4) : 0.4;
  const video = fileInput.files[0];
  const body = new FormData();
  body.append("file", video);

  statusText.textContent =
    mode === "enroll" ? "Enrolling faces from this video..." : "Searching for known faces...";
  grid.innerHTML = "";

  try {
    const response = await fetch(`/detect?mode=${mode}&threshold=${threshold}`, {
      method: "POST",
      body,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Detection failed");
    }

    const { faces, matches_found: matchesFound } = await response.json();
    if (!faces.length) {
      statusText.textContent = "No faces found. Try another video.";
      return;
    }

    if (mode === "enroll") {
      statusText.textContent = `Enrolled ${faces.length} faces from this clip.`;
    } else {
      const matchesNote = matchesFound ? ` (${matchesFound} match${matchesFound === 1 ? "" : "es"})` : "";
      statusText.textContent = `Found ${faces.length} face crops${matchesNote}.`;
    }

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

      if (face.match) {
        const match = document.createElement("div");
        match.className = "match-pill";
        match.textContent = `Match: ${face.match.origin_video} @ ${face.match.origin_timestamp}s (${face.match.similarity_score}% similar)`;
        card.appendChild(match);
      }

      grid.appendChild(card);
    });
  } catch (error) {
    console.error(error);
    statusText.textContent = error.message;
  }
});
