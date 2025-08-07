/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.ui.llmchat

//CUSTOM
import kotlinx.coroutines.flow.StateFlow
//
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.gallery.data.ConfigKey
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.TASK_LLM_CHAT
import com.google.ai.edge.gallery.data.TASK_LLM_ASK_IMAGE
import com.google.ai.edge.gallery.data.Task
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageBenchmarkLlmResult
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageLoading
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageText
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageType
import com.google.ai.edge.gallery.ui.common.chat.ChatMessageWarning
import com.google.ai.edge.gallery.ui.common.chat.ChatSide
import com.google.ai.edge.gallery.ui.common.chat.ChatViewModel
import com.google.ai.edge.gallery.ui.common.chat.Stat
import com.google.ai.edge.gallery.ui.modelmanager.ModelManagerViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

private const val TAG = "AGLlmChatViewModel"
private val STATS = listOf(
  Stat(id = "time_to_first_token", label = "1st token", unit = "sec"),
  Stat(id = "prefill_speed", label = "Prefill speed", unit = "tokens/s"),
  Stat(id = "decode_speed", label = "Decode speed", unit = "tokens/s"),
  Stat(id = "latency", label = "Latency", unit = "sec")
)

private val _customCnnUiResult = MutableStateFlow<String?>(null)
val customCnnUiResult: StateFlow<String?> = _customCnnUiResult



open class LlmChatViewModel(curTask: Task = TASK_LLM_CHAT) : ChatViewModel(task = curTask) {

  fun generateResponse(model: Model, input: String, image: Bitmap? = null, onError: () -> Unit) {
    viewModelScope.launch(Dispatchers.Default) {
      // Declare benchmark-related variables within the coroutine scope
      var firstRun = true
      var start = 0L
      var prefillTokens = 0 // You'll need to calculate this based on the input and model
      var decodeTokens = 0
      var firstTokenTs = 0L
      var timeToFirstToken = 0f
      var prefillSpeed = 0f

      setInProgress(true)
      setPreparing(true)
      start = System.currentTimeMillis() // Initialize start time

      val accelerator = model.getStringConfigValue(key = ConfigKey.ACCELERATOR, defaultValue = "")
      addMessage(
        model = model,
        message = ChatMessageLoading(accelerator = accelerator),
      )

      // Wait for instance to be initialized.
      while (model.instance == null) {
        delay(100)
      }
      delay(500) // Small delay after instance is ready

      // Calculate prefillTokens (example, adapt from your original logic)
      // This needs to happen after model.instance is confirmed non-null
      if (model.instance is LlmModelInstance) { // Check if instance is of expected type
        val instance = model.instance as LlmModelInstance
        prefillTokens = instance.session.sizeInTokens(input) // Assuming sizeInTokens exists
        if (image != null && model.llmSupportImage) { // Check if LLM supports images
          // Add logic to estimate token count for the image if your model/session provides it
          // For example, some models might have a fixed token count for an image input
          // prefillTokens += ESTIMATED_IMAGE_TOKEN_COUNT
          prefillTokens += 257 // Using the placeholder value from your earlier code
        }
      } else {
        Log.w(TAG, "Model instance is not LlmModelInstance, prefillTokens might be inaccurate.")
      }


      try {
        LlmChatModelHelper.runInference(
          llmModel = model,
          input = input,
          image = image,
          llmResultListener = { partialResult, done ->
            val curTs = System.currentTimeMillis()

            if (firstRun) {
              firstTokenTs = System.currentTimeMillis()
              if (start > 0) {
                timeToFirstToken = (firstTokenTs - start) / 1000f
                if (timeToFirstToken > 0f && timeToFirstToken.isFinite()) {
                  prefillSpeed = prefillTokens / timeToFirstToken
                } else {
                  prefillSpeed = 0f
                }
              }
              firstRun = false
              setPreparing(false)
            } else {
              decodeTokens++
            }

            val lastMessage = getLastMessage(model = model)
            if (lastMessage?.type == ChatMessageType.LOADING) {
              removeLastMessage(model = model)
              addMessage(
                model = model,
                message = ChatMessageText(
                  content = "",
                  side = ChatSide.AGENT,
                  accelerator = accelerator
                )
              )
            }

            val latencyMs: Long = if (done && start > 0) System.currentTimeMillis() - start else -1L
            updateLastTextMessageContentIncrementally(
              model = model,
              partialContent = partialResult,
              latencyMs = latencyMs.toFloat()
            )

            if (done) {
              var decodeSpeed = 0f
              if (!firstRun && firstTokenTs > 0) {
                val decodeDurationSeconds = (curTs - firstTokenTs) / 1000f
                if (decodeDurationSeconds > 0f && decodeDurationSeconds.isFinite()) {
                  decodeSpeed = decodeTokens / decodeDurationSeconds
                }
              }
              if (decodeSpeed.isNaN() || decodeSpeed.isInfinite()) decodeSpeed = 0f

              val finalLastMessage = getLastMessage(model = model)
              if (finalLastMessage is ChatMessageText) {
                updateLastTextMessageLlmBenchmarkResult(
                  model = model,
                  llmBenchmarkResult = ChatMessageBenchmarkLlmResult(
                    orderedStats = STATS,
                    statValues = mutableMapOf(
                      "prefill_speed" to prefillSpeed,
                      "decode_speed" to decodeSpeed,
                      "time_to_first_token" to timeToFirstToken,
                      "latency" to if (start > 0) (curTs - start) / 1000f else 0f,
                    ),
                    running = false,
                    latencyMs = -1f,
                    accelerator = accelerator
                  )
                )
              }
            }
          },
          cleanUpListener = {
            setInProgress(false)
            setPreparing(false)
            Log.d(TAG, "LlmChatModelHelper cleanup listener invoked in ViewModel.")
          },
          customCnnResultListener = { cnnOutput ->
            viewModelScope.launch(Dispatchers.Main) {
              Log.d(TAG, "Custom CNN output received in ViewModel: $cnnOutput")
              val resultString = "Custom Model Analysis: ${cnnOutput.toString()}"
              _customCnnUiResult.value = resultString

              addMessage(
                model = model,
                message = ChatMessageText(
                  content = resultString,
                  side = ChatSide.AGENT,
                  accelerator = "CustomCNN"
                )
              )
            }
          }
        )
      } catch (e: Exception) {
        Log.e(TAG, "Error occurred while running inference chain", e)
        setInProgress(false)
        setPreparing(false)
        onError()
      }
    }
  }

  fun stopResponse(model: Model) {
    Log.d(TAG, "Stopping response for model ${model.name}...")
    if (getLastMessage(model = model) is ChatMessageLoading) {
      removeLastMessage(model = model)
    }
    viewModelScope.launch(Dispatchers.Default) {
      setInProgress(false)
      val instance = model.instance as LlmModelInstance
      instance.session.cancelGenerateResponseAsync()
    }
  }

  fun resetSession(model: Model) {
    viewModelScope.launch(Dispatchers.Default) {
      setIsResettingSession(true)
      clearAllMessages(model = model)
      stopResponse(model = model)

      while (true) {
        try {
          LlmChatModelHelper.resetSession(model = model)
          break
        } catch (e: Exception) {
          Log.d(TAG, "Failed to reset session. Trying again")
        }
        delay(200)
      }
      setIsResettingSession(false)
    }
  }

  fun runAgain(model: Model, message: ChatMessageText, onError: () -> Unit) {
    viewModelScope.launch(Dispatchers.Default) {
      // Wait for model to be initialized.
      while (model.instance == null) {
        delay(100)
      }

      // Clone the clicked message and add it.
      addMessage(model = model, message = message.clone())

      // Run inference.
      generateResponse(
        model = model, input = message.content, onError = onError
      )
    }
  }

  fun handleError(
    context: Context,
    model: Model,
    modelManagerViewModel: ModelManagerViewModel,
    triggeredMessage: ChatMessageText,
  ) {
    // Clean up.
    modelManagerViewModel.cleanupModel(task = task, model = model)

    // Remove the "loading" message.
    if (getLastMessage(model = model) is ChatMessageLoading) {
      removeLastMessage(model = model)
    }

    // Remove the last Text message.
    if (getLastMessage(model = model) == triggeredMessage) {
      removeLastMessage(model = model)
    }

    // Add a warning message for re-initializing the session.
    addMessage(
      model = model,
      message = ChatMessageWarning(content = "Error occurred. Re-initializing the session.")
    )

    // Add the triggered message back.
    addMessage(model = model, message = triggeredMessage)

    // Re-initialize the session/engine.
    modelManagerViewModel.initializeModel(
      context = context, task = task, model = model
    )

    // Re-generate the response automatically.
    generateResponse(model = model, input = triggeredMessage.content, onError = {})
  }
}

class LlmAskImageViewModel : LlmChatViewModel(curTask = TASK_LLM_ASK_IMAGE)