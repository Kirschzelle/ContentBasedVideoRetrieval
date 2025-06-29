document.getElementById("reset").addEventListener("click", function () {
    const clipStartFrame = parseInt(this.dataset.clipStartFrame);
    const startFrame = parseInt(this.dataset.startFrame);
    const fps = parseFloat(this.dataset.fps);

    const startMillisecondsDiv = document.getElementById("startMilliseconds");
    startMillisecondsDiv.value = Math.round(((clipStartFrame + startFrame) / fps) * 1000);
});

document.getElementById("submit").addEventListener("click", async function () {
    const submitButton = this;
    const videoId = submitButton.dataset.videoId;

    const startMilliseconds = document.getElementById("startMilliseconds").value;
    const endMilliseconds = document.getElementById("endMilliseconds").value;

    try {
        // LOGIN
        const loginResponse = await fetch("https://vbs.videobrowsing.org/api/v2/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            credentials: "include",
            body: JSON.stringify({
                username: "TECHtalent14",
                password: "gVG7EXSw"
            })
        });

        if (!loginResponse.ok) {
            const error = await loginResponse.json();
            throw new Error(`Login failed: ${error.message || loginResponse.status}`);
        }

        const data = await loginResponse.json();
        console.log('Data:', data);

        const evaluationResponse = await fetch("https://vbs.videobrowsing.org/api/v2/client/evaluation/list", {
            method: "GET",
            credentials: "include"
        });

        if (!evaluationResponse.ok) {
            throw new Error("Failed to fetch evaluation list");
        }

        const evaluationData = await evaluationResponse.json();
        console.log("Evaluations:", evaluationData);
        const evaluation = evaluationData.length ? evaluationData[0] : null;
        if (!evaluation) {
            alert("No active evaluations found");
        }

        const evaluationId = evaluation.id;
        const taskName = evaluation.name;
        console.log("Using evaluationId:", evaluationId, taskName);

        //SUBMIT DATA
        const submitResponse = await fetch(`https://vbs.videobrowsing.org/api/v2/submit/${evaluationId}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            credentials: "include",
            body: JSON.stringify({
                answerSets: [
                    {
                        taskName: taskName,
                        answers: [
                            {
                                text: null,
                                mediaItemName: videoId,
                                mediaItemCollectionName: "IVADL",
                                start: startMilliseconds,
                                end: startMilliseconds
                            }
                        ]
                    }
                ]
            })
        });

        if (!submitResponse.ok) {
            throw new Error("Submission failed!");
        }

        const submitResult = await submitResponse.json();
        console.log("Submit response:", submitResult);

        const resultMessage = document.getElementById("submit-result-message");
        resultMessage.textContent = submitResult.description;
    } catch (err) {
        console.error(err);
        alert("❌ Error submitting to DRES: " + err.message);
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const startMillisecondsInput = document.getElementById("startMilliseconds");
    const video = document.querySelector("video");
    const fps = parseFloat(video.closest(".preview-container").dataset.fps);

    startMillisecondsInput.addEventListener("change", () => {
        const newStartMs = parseInt(startMillisecondsInput.value);
        if (!isNaN(newStartMs)) {
            const newStartTime = newStartMs / 1000;
            video.currentTime = newStartTime;
        }
    });
});