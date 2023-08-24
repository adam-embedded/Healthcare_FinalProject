#include <iostream>

#include <aws/core/Aws.h>
#include <aws/core/client/AsyncCallerContext.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/lexv2-runtime/LexRuntimeV2Client.h>
#include <aws/lexv2-runtime/model/RecognizeUtteranceRequest.h>
#include <aws/lexv2-runtime/model/StartConversationHandler.h>
#include <aws/lexv2-runtime/model/StartConversationRequest.h>
#include <aws/lexv2-runtime/model/StartConversationRequestEventStream.h>

#include <aws/lexv2-runtime/model/RecognizeTextRequest.h>
#include <aws/lexv2-runtime/model/RecognizeTextResult.h>
#include <aws/lexv2-runtime/model/DeleteSessionRequest.h>


using namespace Aws;
using namespace Aws::LexRuntimeV2;

std::string stateString(Model::IntentState state){
    std::string stateStr;
    switch (state) {
        case Model::IntentState::InProgress:
            stateStr = "In Progress";
            break;
        case Model::IntentState::NOT_SET:
            stateStr = "Not Set";
            break;
        case Model::IntentState::Fulfilled:
            stateStr = "Fulfilled";
            break;
        case Model::IntentState::FulfillmentInProgress:
            stateStr = "fulfillment In Progress";
            break;
        case Model::IntentState::ReadyForFulfillment:
            stateStr = "Ready for Fulfillment";
            break;
        case Model::IntentState::Waiting:
            stateStr = "Waiting";
            break;
        case Model::IntentState::Failed:
            stateStr = "failed";
            break;
    }
    return stateStr;
}

void sendText(
        Model::RecognizeTextRequest textRequest,
        std::unique_ptr<Aws::LexRuntimeV2::LexRuntimeV2Client, Aws::Deleter<Aws::LexRuntimeV2::LexRuntimeV2Client>>& lexClient,
        const std::string& value,
        Model::IntentState &state){

    Model::RecognizeTextOutcome textOutcome;
    Model::RecognizeTextResult textResult;

    textRequest.SetText(value);

    textOutcome = lexClient->RecognizeText(textRequest);
    textResult = textOutcome.GetResult();

    auto results = textResult.GetMessages();
    state = textResult.GetSessionState().GetIntent().GetState();

    //printf("State: %d\n", state);
    std::cout << "State: " << stateString(state) << std::endl;


    //here response is given from request
    for (auto result:results){
        std::cout << "Result: " << result.GetContent() << std::endl;
    }
    puts("");
}

int main() {
    //Create Lex runtime object
    std::string lexKey, lexSecret;
    std::string botId, botAliasId, localeId, regionId;

    botId = "ML0DEZ48WH";
    botAliasId = "TSTALIASID";
    lexKey = "";
    lexSecret = "";
    localeId = "en_GB";
    std::string sessionID = "new_session";

    puts("started");

    SDKOptions options;
    options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Debug;
    InitAPI(options);
    Client::ClientConfiguration config;
    //config.region = Region::US_EAST_1;

    auto lexClient = Aws::MakeUnique<LexRuntimeV2Client>("MyClient",
                                                         Aws::Auth::AWSCredentials(lexKey, lexSecret),
                                                         config);

    Model::RecognizeTextRequest textRequest;

    textRequest.SetBotId(botId);
    textRequest.SetBotAliasId(botAliasId);
    textRequest.SetLocaleId(localeId);
    textRequest.SetSessionId(sessionID);

    Model::IntentState state;

    //initiate fall
    sendText(textRequest, lexClient, "i have fallen", state);

    while(state != Model::IntentState::ReadyForFulfillment){
        std::string input;
        std::cin >> input;

        sendText(textRequest, lexClient, input, state);

    }


// close session.
    Model::DeleteSessionRequest deleteSessionRequest;
    deleteSessionRequest.SetSessionId(sessionID);
    deleteSessionRequest.SetLocaleId(localeId);
    deleteSessionRequest.SetBotAliasId(botAliasId);
    deleteSessionRequest.SetBotId(botId);

    lexClient->DeleteSession(deleteSessionRequest);
    Aws::ShutdownAPI(options);

    return 0;
}
