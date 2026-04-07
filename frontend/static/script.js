async function resetEnv() {
  const res = await fetch("/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: 1 })
  });
  const data = await res.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}

async function stepEnv() {
  const res = await fetch("/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action_type: "allocate_jobs",
      amount: 1
    })
  });
  const data = await res.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}

async function getState() {
  const res = await fetch("/state");
  const data = await res.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}

async function gradeEnv() {
  const res = await fetch("/grade", {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  });
  const data = await res.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}

async function getTasks() {
  const res = await fetch("/tasks");
  const data = await res.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}