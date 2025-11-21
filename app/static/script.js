const form = document.getElementById("upload-form");
const fileInput = document.getElementById("video-file");
const statusText = document.getElementById("status");
const grid = document.getElementById("faces-grid");
const thresholdSlider = document.getElementById("match-threshold");
const thresholdValue = document.getElementById("threshold-value");
const resetBtn = document.getElementById("reset-btn");

if (thresholdSlider && thresholdValue) {
  const updateDisplay = () => {
    thresholdValue.textContent = Number(thresholdSlider.value).toFixed(2);
  };
  thresholdSlider.addEventListener("input", updateDisplay);
  updateDisplay();
}

if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    if (!confirm("Clear all enrolled faces from the database?")) {
      return;
    }
    try {
      const response = await fetch("/reset_db", { method: "POST" });
      if (response.ok) {
        statusText.textContent = "✓ Database cleared successfully!";
        grid.innerHTML = "";
      } else {
        statusText.textContent = "Failed to reset database.";
      }
    } catch (error) {
      console.error(error);
      statusText.textContent = "Error resetting database.";
    }
  });
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

    const { faces, matches_found: matchesFound, current_video: currentVideo } = await response.json();
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

      const currentBlock = document.createElement("div");
      currentBlock.className = "face-block";
      const currentImg = document.createElement("img");
      currentImg.src = face.image;
      currentImg.alt = `Face at ${face.timestamp}s`;
      const caption = document.createElement("p");
      caption.innerHTML = `<strong>From: ${currentVideo}</strong><br>${face.timestamp}s`;
      currentBlock.appendChild(currentImg);
      currentBlock.appendChild(caption);
      card.appendChild(currentBlock);

      if (face.match) {
        const matchMeta = document.createElement("div");
        matchMeta.className = "match-pill";
        matchMeta.textContent = `✓ MATCH FOUND (${face.match.similarity_score}% similar)`;
        card.appendChild(matchMeta);

        if (face.match.match_image) {
          const matchBlock = document.createElement("div");
          matchBlock.className = "face-block match-block";
          const matchImg = document.createElement("img");
          matchImg.src = face.match.match_image;
          matchImg.alt = `Matched face from ${face.match.origin_video}`;
          const matchCaption = document.createElement("p");
          matchCaption.innerHTML = `<strong>Matched From: ${face.match.origin_video}</strong><br>${face.match.origin_timestamp}s`;
          matchBlock.appendChild(matchImg);
          matchBlock.appendChild(matchCaption);
          card.appendChild(matchBlock);
        }
      }

      grid.appendChild(card);
    });
  } catch (error) {
    console.error(error);
    statusText.textContent = error.message;
  }
});
