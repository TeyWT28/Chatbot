const msgerForm = get(".msger-inputarea");
const msgerInput = get(".msger-input");
const msgerChat = get(".msger-chat");
// Icons made by Freepik from www.flaticon.com
const BOT_IMG = "./static/girl.png" // "https://image.flaticon.com/icons/svg/327/327779.svg";
const PERSON_IMG = "./static/user.png";
const BOT_NAME = "    ChatBot";
const PERSON_NAME = "You";


$('#textInput').on('input', function(e){
  $('#autocorrect_txt').html("");

    var message = $('#textInput').val();
    console.log(message);
    $.get("/autocorrect",{inputtxt: message}).done(function(data){
        console.log(data);
        if (data!=message){
          document.getElementById("autocorrect_txt").style.visibility = "visible";
          document.getElementById("autocorrect_txt").innerHTML=data;
          $('#autocorrect_txt').html(data);
        }
        
    });
});

$('#autocorrect_txt').on('click', function(e){
  console.log("click")
  //var msg = $('#autocorrect_txt').val();
  var msg = document.getElementById("autocorrect_txt").innerHTML
  console.log(msg);
  msgerInput.value= msg;
  document.getElementById("autocorrect_txt").style.visibility = "hidden";
  $('#autocorrect_txt').html(""); // clear autocorrect txt

});

msgerForm.addEventListener("submit", event => {
  $('#autocorrect_txt').html("");
  event.preventDefault();
  const msgText = msgerInput.value;
  if (!msgText) return;
  appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
  msgerInput.value = "";
  botResponse(msgText);
});
function appendMessage(name, img, side, text) {
  //   Simple solution for small apps
  const msgHTML = `
<div class="msg ${side}-msg">
<div class="msg-img" style="background-image: url(${img})"></div>
<div class="msg-bubble">
<div class="msg-info">
  <div class="msg-info-name">${name}</div>
  <div class="msg-info-time">${formatDate(new Date())}</div>
</div>
<div class="msg-text">${text}</div>
</div>
</div>
`;
  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 500;
}
function botResponse(rawText) {
  // Bot Response
  $.get("/get", { msg: rawText }).done(function (data) {
    console.log(rawText);
    console.log(data);
    const msgText = data;
    appendMessage(BOT_NAME, BOT_IMG, "left", msgText);
  });
}
// Utils
function get(selector, root = document) {
  return root.querySelector(selector);
}
function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();
  return `${h.slice(-2)}:${m.slice(-2)}`;
}