document.getElementById("submit").addEventListener("click", async function () {
    const submitButton = this;
    const videoId = submitButton.dataset.videoId;

    const startFrame = parseInt(submitButton.dataset.startFrame);
    const endFrame = parseInt(submitButton.dataset.endFrame);
    const fps = parseFloat(submitButton.dataset.fps);

    const startMilliseconds = Math.round((startFrame / fps) * 1000);
    const endMilliseconds = Math.round((endFrame / fps) * 1000);

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

        //FETCH EVALUATION SESSIONS
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
                                end: endMilliseconds
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