//
// Created by adam slaymark on 02/08/2023.
//
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cmath>

#ifdef __APPLE__
#include "appleAudio.h"
#else
#include "linuxAudio.h"
#endif

#include <aws/core/Aws.h>
#include <aws/core/client/AsyncCallerContext.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/lexv2-runtime/LexRuntimeV2Client.h>
#include <aws/lexv2-runtime/model/RecognizeUtteranceRequest.h>
#include <aws/lexv2-runtime/model/RecognizeUtteranceResult.h>
#include <aws/lexv2-runtime/model/AudioInputEvent.h>
#include <aws/lexv2-runtime/model/RecognizeTextRequest.h>
#include <aws/lexv2-runtime/model/RecognizeTextResult.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/lexv2-runtime/model/DeleteSessionRequest.h>


#include "soxAudio.h"
#include "vad.h"
#include "lexSpeech.h"

#include <zlib.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>

#include <sndfile.h>
static SF_INFO sfInfo;
static SF_INFO inputInfo;
static SNDFILE * outfile;
static SNDFILE * infile;


using namespace Aws;
using namespace Aws::LexRuntimeV2;


#define RATIO 0.75f
double sampleRate = 16000.0;
const int8_t bitDepth = 16;
const int blocksPerSecond = 50;
const int BlockSize = int(sampleRate / blocksPerSecond);
const int padding_ms = 300;
const int frame_duration_ms = 1000 * BlockSize/sampleRate;
const int num_padding_frames = floor(padding_ms/frame_duration_ms);

static bool triggered = false;

static struct {
    std::deque<std::pair<std::vector<int16_t>,bool>> circ_buff;
    std::mutex lock;
}audio_adc;

static struct {
    std::vector<std::vector<int16_t>> frames;
    std::mutex lock;
    bool execute = false;
}speechBuf;


//this is a threaded callback function for audio, this can be from linux and apple
// Callback is triggered when buffer is full
void callbackAbstraction(const int16_t * inputBuffer, uint32_t bufferSize, void* userdata){

    auto complete =  static_cast<bool*>(userdata);

    if (bufferSize < (BlockSize*2)){
        fprintf(stderr,"Buffer Under run\n");
        return;
    }
    //input buffer is thread locked within this abstraction. it will be re-attached within the driver.

    int division = bitDepth/8;

    speechBuf.lock.lock();

    //speechBuf.lock.unlock();

    if (speechBuf.execute) {
        //speechBuf.lock.unlock();

        //puts("made it in...");

        //if speech detected append vector
        int speech = is_speech(inputBuffer, bufferSize / division);


        //printf("||Is Speech: %d \n", speech);

        if (speech < 0) {
            fprintf(stderr, "VAD processing failed\n");
            return;
        }

        //write array to vector
        std::vector<int16_t> tmpVec;
        tmpVec.insert(tmpVec.end(), inputBuffer, inputBuffer + (bufferSize / division));

        if (!triggered) {
            //lock variable to thread to stop race conditions
            audio_adc.lock.lock();
            //Check if buffer has exceeded size, if it has, remove a sample before adding another.
            if (audio_adc.circ_buff.size() == num_padding_frames) {
                audio_adc.circ_buff.pop_front();
            }
            //write to circular buffer
            //add to buffer and mark with speech value
            audio_adc.circ_buff.emplace_back(tmpVec, bool(speech));

            int num_voiced = 0;
            for (auto value: audio_adc.circ_buff) {
                if (value.second) num_voiced++;
            }
            //if (num_voiced > 1) std::cout << "Num Voiced: " << num_voiced << std::endl;

            if (num_voiced > (RATIO * float(num_padding_frames))) {
                triggered = true;
                //lock deepstream buffer
                //speechBuf.lock.lock();
                for (auto &i: audio_adc.circ_buff) {
                    speechBuf.frames.push_back(i.first);
                }
                //speechBuf.lock.unlock();

                //clear the circular buffer
                audio_adc.circ_buff.clear();
            }
            audio_adc.lock.unlock();
            speechBuf.lock.unlock();
        } else {
            //Place frames within deepstream buffer
            //speechBuf.lock.lock();
            speechBuf.frames.emplace_back(tmpVec);


            audio_adc.lock.lock();
            audio_adc.circ_buff.emplace_back(tmpVec, speech);

            int num_not_voiced = 0;
            for (const auto &value: audio_adc.circ_buff) {
                if (!value.second) num_not_voiced++;
            }
            if (num_not_voiced > (RATIO * float(num_padding_frames))) {
                triggered = false;
                audio_adc.circ_buff.clear();
                puts("speech identified");
                speechBuf.execute = false;
                *complete = true;
            }
            audio_adc.lock.unlock();
            speechBuf.lock.unlock();
        }
    } else {
        speechBuf.lock.unlock();
    };

}

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
        Model::RecognizeTextRequest &textRequest,
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

void sendSound(Model::RecognizeUtteranceRequest &utteranceRequest,
               std::unique_ptr<Aws::LexRuntimeV2::LexRuntimeV2Client, Aws::Deleter<Aws::LexRuntimeV2::LexRuntimeV2Client>>& lexClient,
               const std::string & value,
               Model::IntentState &state,
               char* buffer,
               const uint32_t bufferSize){

    Model::RecognizeUtteranceOutcome utteranceOutcome;
    Model::RecognizeUtteranceResult utteranceResult;



    utteranceRequest.SetRequestContentType("audio/l16; rate=16000; channels=1");
    utteranceRequest.SetResponseContentType("audio/pcm");

    std::stringstream stream;
    std::shared_ptr<std::basic_iostream<char, std::char_traits<char>>> sharedStream = std::make_shared<std::stringstream>(std::move(stream));


    sharedStream->write(buffer, bufferSize);

    utteranceRequest.SetBody(sharedStream);
    utteranceRequest.SetContentType("audio/l16; rate=16000; channels=1");


    std::string trans;

    utteranceOutcome = lexClient->RecognizeUtterance(utteranceRequest);

    if (utteranceOutcome.IsSuccess()) {

        utteranceResult = utteranceOutcome.GetResultWithOwnership();

        //std::string resultText = utteranceResult.GetMessages();
        auto resultAudioType = utteranceResult.GetContentType();

        Aws::IOStream &resultAudio = utteranceResult.GetAudioStream();

        auto sessionState = utteranceResult.GetSessionState();


        //copy in to buffer. however this could be streamed directly to the sound card in 1024 byte chunks
        // You cannot run both examples at the same time as read empties the stream.
        //if you want to load into char buffer with whole audio response:
        if (resultAudio) {
            //get length of file
            resultAudio.seekg(0, resultAudio.end);
            int length = resultAudio.tellg();
            resultAudio.seekg(0, resultAudio.beg);

            char *tmpbuffer = new char[length];
            resultAudio.read(tmpbuffer, length);


            if (length == 0){
                puts("empty Buffer");
                return;
            }

            // write response to file
            //sf_write_raw(outfile, tmpbuffer, length);
            //writeAudioSox(tmpbuffer, length);

            PlayAudio(tmpbuffer, length);

            //int len = std::ceil(length / (sampleRate * 2));
            //std::cout << "length: " << len <<std::endl;

            //std::this_thread::sleep_for(std::chrono::seconds(len));

            puts("Response Played");
        }

        resultAudio.clear();



//        //we need to decompress message to be human-readable
        std::string based = Base64Decode(sessionState);
        std::string decompressedData = GzipDecompress(based);

        //auto tmpState = static_cast<Model::IntentState>(decompressedData);

        std::cout << "temporary state: " << decompressedData << std::endl;
    }
}

void GetSound(Model::RecognizeUtteranceRequest &utteranceRequest,
              std::unique_ptr<Aws::LexRuntimeV2::LexRuntimeV2Client, Aws::Deleter<Aws::LexRuntimeV2::LexRuntimeV2Client>>& lexClient){
    Model::RecognizeUtteranceOutcome utteranceOutcome;

    utteranceRequest.SetRequestContentType("audio/l16; rate=16000; channels=1");
    utteranceRequest.SetResponseContentType("audio/pcm");
    utteranceRequest.SetContentType("audio/l16; rate=16000; channels=1");
    utteranceOutcome = lexClient->RecognizeUtterance(utteranceRequest);

    if (utteranceOutcome.IsSuccess()) {
        Aws::IOStream &resultAudio = utteranceOutcome.GetResultWithOwnership().GetAudioStream();

//        if (resultAudio) {
//            while (!resultAudio.eof()) {
//                char tmpBuf[1024];
//                resultAudio.read(tmpBuf, sizeof(tmpBuf)); // read from stream
//                auto bytesRead = resultAudio.gcount(); // check how much was read, you need this to know how much to send
//
//                //Push to audio card here...
//                //writeAudioSox(tmpBuf, bytesRead);
//                PlayAudio(tmpBuf, bytesRead);
//
//            }
//        }


        if (resultAudio) {
            //get length of file
            resultAudio.seekg(0, resultAudio.end);
            int length = resultAudio.tellg();
            resultAudio.seekg(0, resultAudio.beg);


            char *tmpbuffer = new char[length];
            resultAudio.read(tmpbuffer, length);

            // write response to file
            //sf_write_raw(outfile, tmpbuffer, length);
            //writeAudioSox(tmpbuffer, length);

            PlayAudio(tmpbuffer, length);
//            int len = std::ceil(length / (sampleRate*2));
//            //std::cout << "length: " << len <<std::endl;
//
//            std::this_thread::sleep_for(std::chrono::seconds(len));
        }

        resultAudio.clear();
    } else puts("Failed get audio");

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
    std::string sessionID = "new_session1";

    puts("started");

    initAudio(uint32_t(sampleRate), bitDepth, BlockSize,1024);
    //initSox(int(sampleRate), bitDepth);
    initVAD(sampleRate, 3);

    SDKOptions options;
    options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Debug;
    InitAPI(options);
    Client::ClientConfiguration config;
    //config.region = Region::US_EAST_1;

    auto lexClient = Aws::MakeUnique<LexRuntimeV2Client>("MyClient",
                                                         Aws::Auth::AWSCredentials(lexKey, lexSecret),
                                                         config);

    Model::RecognizeTextRequest textRequest;
    Model::RecognizeUtteranceRequest utteranceRequest;

    textRequest.SetBotId(botId);
    textRequest.SetBotAliasId(botAliasId);
    textRequest.SetLocaleId(localeId);
    textRequest.SetSessionId(sessionID);

    utteranceRequest.SetBotId(botId);
    utteranceRequest.SetBotAliasId(botAliasId);
    utteranceRequest.SetLocaleId(localeId);
    utteranceRequest.SetSessionId(sessionID);

    //startAudioStream();

    Model::IntentState state;

    const char* outputFilename = "VAD_recorded_audio.wav";
    sfInfo.channels = 1;
    sfInfo.samplerate = 16000;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    outfile = sf_open(outputFilename, SFM_WRITE, &sfInfo);
    if (!outfile) {
        std::cerr << "Error opening the output WAV file.\n";
        return 1;
    }

    //read audio file for test
    const char* infilename = "yes1.wav";
    infile = sf_open(infilename, SFM_READ, &inputInfo);
    if (!infile) {
        std::cerr << "Error opening the input WAV file.\n";
        return 1;
    }
    char* audioData = (char *) malloc(inputInfo.frames * inputInfo.channels * sizeof(char));
    if (!audioData){
        printf("Memory allocation failed\n");
        sf_close(infile);
        return 1;
    }
    sf_read_raw(infile, (char*)audioData, inputInfo.frames);

    sf_close(infile);


    //initiate fall
    sendText(textRequest, lexClient, "i have fallen", state);

    GetSound(utteranceRequest, lexClient);

    //std::this_thread::sleep_for(std::chrono::milliseconds(500));

    //sendSound(utteranceRequest, lexClient, "", state, audioData, inputInfo.frames * inputInfo.channels * sizeof(char));

    free(audioData);

//    int len = std::ceil((inputInfo.frames * inputInfo.channels * sizeof(char)) / sampleRate);
//    std::cout << "length: " << len <<std::endl;
//
//    std::this_thread::sleep_for(std::chrono::seconds(len));

    //puts("past Halt");

    //startAudioStream();

    //std::this_thread::sleep_for(std::chrono::milliseconds(500)); // wait for audio to warm up

    // Execute audio sampling
    speechBuf.lock.lock();
    speechBuf.execute = true;
    speechBuf.lock.unlock();

    puts("ready for input");

    while(state != Model::IntentState::ReadyForFulfillment){
//        std::string input;
//        std::cin >> input;
//
//        sendText(textRequest, lexClient, input, state);

        //collect sound
        RecordAudio();

        speechBuf.lock.lock();
        if (!speechBuf.execute) {

            //compile all frames in to 1 vector
            std::vector<int16_t> tmpVec;
            for (auto frame: speechBuf.frames){
                tmpVec.insert(tmpVec.end(), frame.begin(),frame.end());
            }

            speechBuf.frames.clear();

            // convert vector in to bytes array of fixed length
            char bytesArray[tmpVec.size()*2];
            uint32_t bufferSize = tmpVec.size() * sizeof(int16_t);
            std::memcpy(bytesArray, tmpVec.data(),bufferSize);

            //sf_write_raw(outfile, bytesArray, bufferSize*2);


            // convert vector in to array of fixed length
            int16_t deepBuf[tmpVec.size()];
            for (int i=0;i< (sizeof(deepBuf) / sizeof(deepBuf[0]));i++){
                deepBuf[i] = tmpVec[i];
            }

            //uint32_t bufferSize1 = (sizeof(deepBuf) / sizeof(deepBuf[0]));
            //std::string str;

            //sf_write_short(outfile,deepBuf,bufferSize1);


            sendSound(utteranceRequest, lexClient, "", state, bytesArray, bufferSize);

            state = textRequest.GetSessionState().GetIntent().GetState();

            std::cout << "State: " << stateString(state) << std::endl;

            speechBuf.execute = true;
            speechBuf.lock.unlock();

            puts("finished transaction\n");

        } else speechBuf.lock.unlock();


    }


// close session.
    Model::DeleteSessionRequest deleteSessionRequest;
    deleteSessionRequest.SetSessionId(sessionID);
    deleteSessionRequest.SetLocaleId(localeId);
    deleteSessionRequest.SetBotAliasId(botAliasId);
    deleteSessionRequest.SetBotId(botId);

    lexClient->DeleteSession(deleteSessionRequest);
    Aws::ShutdownAPI(options);


    //closeSox();
#ifdef __APPLE__
    stopAudioStream();
#endif
    return 0;
}
