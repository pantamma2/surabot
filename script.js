$(document).ready(function() {
    $("#send-button").click(function() {
        sendMessage();
    });

    $("#user-input").keypress(function(event) {
        if (event.which == 13) {
            sendMessage();
        }
    });

    function sendMessage() {
        var userMessage = $("#user-input").val().trim();
        if (userMessage === "") return;

        $("#chat-box").append('<div class="message user-message">' + userMessage + '</div>');
        $("#user-input").val("");

        $.ajax({
            type: "POST",
            url: "/send_message",
            contentType: "application/json",
            data: JSON.stringify({ message: userMessage }),
            success: function(response) {
                $("#chat-box").append('<div class="message bot-message">' + response + '</div>');
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            },
            error: function(xhr, status, error) {
                alert("Error: " + error);
            }
        });
    }
});
