
class BaseChatbot:
    def __init__(self):
        '''
        Initialize the model and other things here
        '''


    def run(self, input_text):
        '''
        Yields the response to the input text
        '''
        raise NotImplementedError("This method should be implemented by the subclass")

    def post_process(self, response):
        '''
        Post process the response before returning
        '''
        return response

    def generate_response(self, input_text):
        out = self.run(input_text)
        response_text = ''
        for o in out:
            text = o
            response_text += text
        response = self.post_process(response_text)
        return response

    def generate_response_stream(self, input_text, output_queue, interrupt_queue):
        out = self.run(input_text)
        response_text = ''
        for o in out:
            if not interrupt_queue.empty():
                break
            text = o
            output_queue.put(text)
            response_text += text
        output_queue.put(None)
        response = self.post_process(response_text)
        return response
