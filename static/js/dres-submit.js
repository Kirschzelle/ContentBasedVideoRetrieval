document.getElementById("submit").addEventListener("click", async function () {
    const button = this;

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
        /*const submitResponse = await fetch(`https://vbs.videobrowsing.org/api/v2/submit/${evaluationId}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            credentials: "include",
            body: JSON.stringify({
                answerSets: [
                    {
                        // taskId or taskName are optional — inferred if not provided
                        answers: [
                            {
                                text: null, // or a string if needed
                                mediaItemName: "your-video-id", // <== change this!
                                mediaItemCollectionName: "IVADL", // required if not inferred
                                start: 10000,  // in milliseconds
                                end: 10000     // same as start if it's a single point
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
        console.log("Submit response:", submitResult);*/
    } catch (err) {
        console.error(err);
        alert("❌ Error submitting to DRES: " + err.message);
    }
});